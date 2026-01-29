"""
语义切片器
基于语义相似度在话题切换处自动断句
"""
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ability.operators.chunkers.base_chunker import BaseChunker, Chunk
from ability.utils.text_processing import split_by_sentences


class SemanticChunker(BaseChunker):
    """语义切片器，基于语义相似度自动断句"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化语义切片器

        Args:
            config: 配置字典
                - similarity_threshold: 相似度阈值（默认0.7）
                - min_chunk_size: 最小块大小（字符数，默认100）
                - max_chunk_size: 最大块大小（字符数，默认1000）
                - model_name: 语义模型名称（可选）
        """
        super().__init__(config)
        self.similarity_threshold = self.get_config("similarity_threshold", 0.7)
        self.min_chunk_size = self.get_config("min_chunk_size", 100)
        self.max_chunk_size = self.get_config("max_chunk_size", 1000)
        self.model_name = self.get_config("model_name", "BAAI/bge-small-zh-v1.5")
        self.model: Optional[SentenceTransformer] = None
        self._fallback_enabled: bool = False

    def _initialize(self) -> None:
        """初始化语义模型"""
        if SentenceTransformer is None:
            self._fallback_enabled = True
            self.logger.warning(
                "sentence-transformers is not installed; semantic chunker will fall back to rule-based splitting."
            )
            return

        self.logger.info(f"Loading semantic model: {self.model_name}")
        try:
            # 优先使用本地缓存，避免在无网络/受限环境下触发 HuggingFace 重试等待
            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except TypeError:
                # sentence-transformers 版本不支持 local_files_only 时，直接降级（避免网络请求）
                raise RuntimeError("SentenceTransformer does not support local_files_only")  # noqa: TRY301
            self.logger.info("Semantic model loaded")
        except Exception as e:  # noqa: BLE001
            self.model = None
            self._fallback_enabled = True
            self.logger.warning(
                f"Failed to load semantic model '{self.model_name}', falling back to rule-based splitting. Error: {e}"
            )

    def _fallback_chunk_by_sentences(self, text: str, **kwargs) -> List[Chunk]:
        """离线/无模型时的降级切片：按句子聚合到 max_chunk_size。"""
        # 按句子分割
        sentence_delimiters = kwargs.get("sentence_delimiters")
        sentences = split_by_sentences(text, sentence_delimiters=sentence_delimiters)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current = ""
        chunk_index = 0
        start_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = (current + " " + sentence).strip() if current else sentence
            if current and len(candidate) > self.max_chunk_size:
                chunks.append(
                    Chunk(
                        content=current,
                        chunk_index=chunk_index,
                        start_index=start_index,
                        end_index=start_index + len(current),
                        metadata={
                            "strategy": "semantic",
                            "fallback": True,
                        },
                    )
                )
                start_index = start_index + len(current)
                chunk_index += 1
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(
                Chunk(
                    content=current,
                    chunk_index=chunk_index,
                    start_index=start_index,
                    end_index=start_index + len(current),
                    metadata={
                        "strategy": "semantic",
                        "fallback": True,
                    },
                )
            )

        return chunks

    def _chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        基于语义相似度切片

        Args:
            text: 输入文本
            **kwargs: 额外的处理参数

        Returns:
            文档块列表
        """
        text = self._clean_text(text, **kwargs)

        # 无法加载语义模型（离线/未安装等）时，降级为规则切片，确保服务可用 & 测试稳定
        if not self.model:
            return self._fallback_chunk_by_sentences(text, **kwargs)

        # 按句子分割
        sentence_delimiters = kwargs.get("sentence_delimiters")
        sentences = split_by_sentences(text, sentence_delimiters=sentence_delimiters)

        if not sentences:
            return []

        # 如果文本较短，直接返回单个块
        if len(text) <= self.max_chunk_size:
            return [
                Chunk(
                    content=text,
                    chunk_index=0,
                    start_index=0,
                    end_index=len(text),
                    metadata={
                        "strategy": "semantic",
                        "sentence_count": len(sentences),
                    },
                )
            ]

        chunks = []
        current_chunk_sentences = []
        current_chunk_text = ""
        chunk_index = 0
        start_index = 0

        for i, sentence in enumerate(sentences):
            # 如果添加当前句子会超过最大大小，先处理当前块
            if current_chunk_text and len(current_chunk_text + sentence) > self.max_chunk_size:
                # 创建块
                chunk = Chunk(
                    content=current_chunk_text.strip(),
                    chunk_index=chunk_index,
                    start_index=start_index,
                    end_index=start_index + len(current_chunk_text),
                    metadata={
                        "strategy": "semantic",
                        "sentence_count": len(current_chunk_sentences),
                    },
                )
                chunks.append(chunk)

                # 重置
                current_chunk_sentences = []
                current_chunk_text = ""
                start_index = chunk.end_index
                chunk_index += 1

            # 添加当前句子
            current_chunk_sentences.append(sentence)
            current_chunk_text += sentence + " "

            # 如果当前块达到最小大小，检查是否可以分割
            if len(current_chunk_text) >= self.min_chunk_size and len(current_chunk_sentences) > 1:
                # 计算最后两个句子的相似度
                if len(current_chunk_sentences) >= 2:
                    prev_sentence = current_chunk_sentences[-2]
                    curr_sentence = current_chunk_sentences[-1]

                    # 计算相似度
                    embeddings = self.model.encode([prev_sentence, curr_sentence])
                    similarity = self._cosine_similarity(embeddings[0], embeddings[1])

                    # 如果相似度低于阈值，在当前句子前分割
                    if similarity < self.similarity_threshold:
                        # 创建块（不包含当前句子）
                        chunk_text = " ".join(current_chunk_sentences[:-1])
                        chunk = Chunk(
                            content=chunk_text.strip(),
                            chunk_index=chunk_index,
                            start_index=start_index,
                            end_index=start_index + len(chunk_text),
                            metadata={
                                "strategy": "semantic",
                                "sentence_count": len(current_chunk_sentences) - 1,
                                "similarity": float(similarity),
                            },
                        )
                        chunks.append(chunk)

                        # 重置（保留当前句子）
                        start_index = chunk.end_index
                        chunk_index += 1
                        current_chunk_sentences = [sentence]
                        current_chunk_text = sentence + " "

        # 处理最后一个块
        if current_chunk_text.strip():
            chunk = Chunk(
                content=current_chunk_text.strip(),
                chunk_index=chunk_index,
                start_index=start_index,
                end_index=start_index + len(current_chunk_text),
                metadata={
                    "strategy": "semantic",
                    "sentence_count": len(current_chunk_sentences),
                },
            )
            chunks.append(chunk)

        return chunks

    def _cosine_similarity(self, vec1, vec2) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度值（0-1）
        """
        import numpy as np

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
