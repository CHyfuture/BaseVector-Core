"""
日志系统
"""
import sys
from pathlib import Path

from loguru import logger

from ability.config import get_settings

settings = get_settings()


def setup_logger():
    """配置日志系统"""
    # 移除默认处理器
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # 文件输出
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level=settings.LOG_LEVEL,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # 错误日志单独记录
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="ERROR",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    return logger


# 初始化日志
setup_logger()
