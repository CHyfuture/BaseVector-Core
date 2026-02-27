"""
文档解析相关异常
"""


class ParseError(Exception):
    """解析失败（通用）"""
    pass


class DocTimeoutError(ParseError):
    """文档解析/转换超时"""
    pass


class NetworkError(ParseError):
    """网络连接失败"""
    pass


class APIError(ParseError):
    """外部 API 调用错误（如 HTTP 非 2xx）"""
    pass
