"""
基于 MinerU 的文档解析器：支持 PDF / Word / PPT
非 PDF 先经 LibreOffice 转为 PDF，再统一由 MinerU 解析为 Markdown
"""
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ability.config import get_settings
from ability.operators.parsers.base_parser import BaseParser
from ability.operators.parsers.exceptions import (
    APIError,
    DocTimeoutError,
    NetworkError,
    ParseError,
)


# 进程内锁，限制并发调用 MinerU（多进程部署可改为 Redis 锁）
_mineru_lock = threading.Lock()


class MinerUParser(BaseParser):
    """
    使用 MinerU API 解析 PDF/Word/PPT，返回 Markdown。
    对 .doc/.docx/.ppt/.pptx 先调用 LibreOffice 转为 PDF，再交给 MinerU。
    """

    # 需要先转 PDF 的格式
    OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_extensions = [".pdf", ".doc", ".docx", ".ppt", ".pptx"]
        settings = get_settings()
        self.mineru_url = (config or {}).get("mineru_url") or getattr(
            settings, "MINERU_URL", "http://localhost:8000"
        )
        self.mineru_backend = (config or {}).get("mineru_backend") or getattr(
            settings, "MINERU_BACKEND", "pdf"
        )
        self.mineru_parse_method = (config or {}).get("mineru_parse_method") or getattr(
            settings, "MINERU_PARSE_METHOD", "auto"
        )
        self.lock_timeout = (config or {}).get("lock_timeout") or getattr(
            settings, "MINERU_LOCK_TIMEOUT", 3600
        )
        self.lock_wait_timeout = (config or {}).get("lock_wait_timeout") or getattr(
            settings, "MINERU_LOCK_WAIT_TIMEOUT", 600
        )
        self.request_timeout = (config or {}).get("request_timeout") or getattr(
            settings, "MINERU_REQUEST_TIMEOUT", 3000
        )
        self.libreoffice_timeout = (config or {}).get("libreoffice_convert_timeout") or getattr(
            settings, "LIBREOFFICE_CONVERT_TIMEOUT", 3000
        )

    def _find_content(self, data: Any) -> Optional[str]:
        """
        递归地在嵌套的字典或列表中查找 'md_content' 的值
        """
        if isinstance(data, dict):
            if "md_content" in data:
                content = data["md_content"]
                if isinstance(content, str):
                    return content
            for value in data.values():
                result = self._find_content(value)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_content(item)
                if result is not None:
                    return result
        return None

    def _convert_to_pdf_with_libreoffice(self, file_path: str, output_dir: str) -> str:
        """
        使用 LibreOffice 无头模式转换为 PDF。
        注意：运行环境需安装 libreoffice（soffice 或 libreoffice 命令）。
        """
        user_profile_dir = tempfile.mkdtemp(prefix="lo_profile_")
        try:
            cmd = [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", output_dir,
                f"-env:UserInstallation=file://{user_profile_dir}",
                file_path,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.libreoffice_timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"LibreOffice 转换失败: {result.stderr.decode(errors='ignore')}"
                )

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            pdf_path = os.path.join(output_dir, base_name + ".pdf")
            if not os.path.exists(pdf_path):
                for name in os.listdir(output_dir):
                    if name.lower().endswith(".pdf"):
                        return os.path.join(output_dir, name)
                raise RuntimeError("LibreOffice 转换完成但未找到 PDF 输出")
            return pdf_path
        finally:
            try:
                shutil.rmtree(user_profile_dir)
            except Exception:
                pass

    def _parse_with_mineru(self, file_path: str, file_name: str) -> str:
        """调用 MinerU API 解析 PDF，返回 Markdown 文本。"""
        acquired = _mineru_lock.acquire(blocking=True, timeout=self.lock_wait_timeout)
        if not acquired:
            raise DocTimeoutError("等待 MinerU 解析锁超时")

        try:
            params = {
                "output_dir": "",
                "lang_list": ["ch"],
                "backend": self.mineru_backend,
                "parse_method": self.mineru_parse_method,
                "formula_enable": True,
                "table_enable": True,
                "return_md": True,
                "return_middle_json": False,
                "return_model_output": False,
                "return_content_list": False,
                "return_images": False,
                "start_page_id": 0,
                "end_page_id": 99999,
            }

            with open(file_path, "rb") as f:
                files = [("files", (file_name, f, "application/pdf"))]
                resp = requests.post(
                    f"{self.mineru_url.rstrip('/')}/file_parse",
                    files=files,
                    data=params,
                    timeout=self.request_timeout,
                )
            resp.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise DocTimeoutError(f"MinerU 调用超时: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"MinerU 网络连接失败: {e}") from e
        except requests.exceptions.HTTPError as e:
            raise APIError(f"MinerU HTTP 错误: {e}") from e
        except (DocTimeoutError, NetworkError, APIError):
            raise
        except Exception as e:
            raise ParseError(f"MinerU 调用失败: {e}") from e
        finally:
            try:
                _mineru_lock.release()
            except Exception:
                pass

        data = resp.json()
        content = self._find_content(data)
        if content is None:
            raise ParseError("MinerU 返回中未找到 md_content")
        return content

    def _parse(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        解析文档：非 PDF 先转 PDF，再统一用 MinerU 解析为 Markdown。
        """
        path_str = str(file_path)
        suffix = file_path.suffix.lower()
        file_name = file_path.name

        if suffix in self.OFFICE_EXTENSIONS:
            output_dir = tempfile.mkdtemp(prefix="mineru_convert_")
            try:
                pdf_path = self._convert_to_pdf_with_libreoffice(path_str, output_dir)
                content = self._parse_with_mineru(pdf_path, os.path.basename(pdf_path))
            finally:
                try:
                    shutil.rmtree(output_dir)
                except Exception:
                    pass
        else:
            content = self._parse_with_mineru(path_str, file_name)

        return {
            "content": content,
            "metadata": {"source": file_name},
            "structure": {},
        }
