"""压缩包处理组件：识别并内存中只读解析 .zip/.tar/.tar.gz/.tgz/.jar/.war。

功能：
- 识别 .zip/.tar/.tar.gz/.tgz/.jar/.war 格式
- 内存中只读解析，不解压到磁盘
- 50MB 压缩包限制
- 5MB 单文件限制
- 加密/损坏压缩包跳过并记录
- 生成压缩包内文件索引
"""

import io
import os
import struct
import zlib
import tarfile
import zipfile
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path


class CompressionError(Exception):
    """压缩包处理错误。"""
    pass


MAX_PACKAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

SUPPORTED_ARCHIVE_EXTS = ('.zip', '.tar', '.tar.gz', '.tgz', '.jar', '.war')


class CompressionPackage:
    """压缩包表示，仅在内存中操作。"""

    MAX_PACKAGE_SIZE = MAX_PACKAGE_SIZE
    MAX_FILE_SIZE = MAX_FILE_SIZE
    SUPPORTED_EXTENSIONS = set(SUPPORTED_ARCHIVE_EXTS)

    def __init__(self, data: bytes, origin_name: str = ""):
        self.origin_name = origin_name
        self.data = data
        self.file_index: Dict[str, dict] = {}
        self.is_valid = False
        self.error_reason: Optional[str] = None
        self.skipped: List[Dict[str, str]] = []
        self.detected_format: Optional[str] = None

        self._parse()

    def _skip(self, path: str, reason: str):
        self.skipped.append({'path': path, 'reason': reason})

    def _parse(self):
        """解析压缩包（内存中，不写入磁盘）。"""
        size = len(self.data)

        if size > self.MAX_PACKAGE_SIZE:
            self.error_reason = (
                f"压缩包过大: {size / (1024*1024):.1f}MB > "
                f"{self.MAX_PACKAGE_SIZE/(1024*1024):.0f}MB"
            )
            return

        ext = self._detect_extension()
        if ext is None:
            self.error_reason = "未识别的压缩包格式"
            return
        self.detected_format = ext

        try:
            if ext in ('.zip', '.jar', '.war'):
                self._parse_zip()
            elif ext in ('.tar', '.tar.gz', '.tgz'):
                self._parse_tar()
            else:
                self.error_reason = f"不支持的格式: {ext}"
                return
            self.is_valid = True
        except CompressionError as e:
            self.error_reason = str(e)
        except Exception as e:
            self.error_reason = f"压缩包解析错误: {e}"

    def _detect_extension(self) -> Optional[str]:
        """按文件名扩展名 + 魔数检测压缩包格式。"""
        name = (self.origin_name or "").lower()
        for candidate in ('.tar.gz', '.tgz', '.zip', '.tar', '.jar', '.war'):
            if name.endswith(candidate):
                return candidate

        head = self.data[:8]
        if head[:2] == b'\x1f\x8b':
            return '.tar.gz'
        if head[:4] in (b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'):
            if name.endswith('.jar'):
                return '.jar'
            if name.endswith('.war'):
                return '.war'
            return '.zip'
        if len(self.data) > 265 and self.data[257:262] == b'ustar':
            return '.tar'
        if head[:7] == b'Rar!\x1a\x07\x00':
            return None
        return None

    def _parse_zip(self):
        """解析 ZIP 格式（包括 jar/war）。只读，不解压到磁盘。"""
        from atomic_io import open_blob
        # BytesIO 分级：小包直接 BytesIO，大包落临时文件，避免再复制一份缓冲区
        with open_blob(self.data, suffix=".zip") as stream:
            self._parse_zip_stream(stream)

    def _parse_zip_stream(self, stream):
        try:
            zf = zipfile.ZipFile(stream, 'r')
        except (zipfile.BadZipFile, RuntimeError, OSError) as e:
            raise CompressionError(f"无效的 ZIP 格式: {e}")

        try:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel_path = info.filename.lstrip('/\\').replace('\\', '/')

                if info.flag_bits & 0x1:
                    self._skip(rel_path, "加密条目")
                    continue
                if info.file_size > self.MAX_FILE_SIZE:
                    self._skip(
                        rel_path,
                        f"单文件超限 {info.file_size} > {self.MAX_FILE_SIZE}",
                    )
                    continue

                try:
                    content = zf.read(info.filename)
                except RuntimeError as e:
                    reason = "加密条目" if "password" in str(e).lower() or "encrypted" in str(e).lower() else f"读取失败: {e}"
                    self._skip(rel_path, reason)
                    continue
                except (zipfile.BadZipFile, OSError, EOFError) as e:
                    self._skip(rel_path, f"损坏条目: {e}")
                    continue

                self.file_index[rel_path] = {
                    'path': rel_path,
                    'size': info.file_size,
                    'compressed_size': info.compress_size,
                    'mtime': info.date_time,
                    'content': content,
                    'compression_method': info.compress_type,
                }
        finally:
            zf.close()

    def _parse_tar(self):
        """解析 tar 格式（.tar, .tar.gz, .tgz）。只读，不解压到磁盘。"""
        from atomic_io import open_blob
        with open_blob(self.data, suffix=".tar") as stream:
            self._parse_tar_stream(stream)

    def _parse_tar_stream(self, stream):
        try:
            tf = tarfile.open(fileobj=stream, mode='r:*')
        except (tarfile.TarError, EOFError, OSError) as e:
            raise CompressionError(f"无效的 tar 格式: {e}")

        try:
            for member in tf.getmembers():
                if not member.isfile():
                    if member.isdir():
                        continue
                    self._skip(member.name, f"非普通文件类型 {member.type!r}")
                    continue

                rel_path = member.name.lstrip('/\\').replace('\\', '/')

                if member.size > self.MAX_FILE_SIZE:
                    self._skip(
                        rel_path,
                        f"单文件超限 {member.size} > {self.MAX_FILE_SIZE}",
                    )
                    continue

                try:
                    f = tf.extractfile(member)
                    if f is None:
                        self._skip(rel_path, "无法读取条目")
                        continue
                    content = f.read()
                except (tarfile.ExtractError, OSError, EOFError) as e:
                    self._skip(rel_path, f"损坏条目: {e}")
                    continue

                self.file_index[rel_path] = {
                    'path': rel_path,
                    'size': member.size,
                    'mtime': member.mtime,
                    'content': content,
                }
        finally:
            tf.close()
    
    def get_file(self, rel_path: str) -> Optional[dict]:
        """获取文件信息。
        
        Args:
            rel_path: 相对于压缩包根目录的相对路径
            
        Returns:
            包含文件信息的字典，如果未找到则返回 None
        """
        # 标准化路径
        norm_path = rel_path.lstrip('/\\').replace('\\', '/')
        return self.file_index.get(norm_path)
    
    def get_file_content(self, rel_path: str) -> Optional[bytes]:
        """获取文件内容（二进制）。
        
        Args:
            rel_path: 相对于压缩包根目录的相对路径
            
        Returns:
            文件的二进制内容，如果未找到则返回 None
        """
        file_info = self.get_file(rel_path)
        if file_info:
            return file_info.get('content')
        return None
    
    def get_file_index(self) -> Dict[str, dict]:
        """获取压缩包内文件索引。
        
        Returns:
            文件路径到元数据的映射
        """
        return dict(self.file_index)
    
    def get_file_list(self) -> List[str]:
        """获取压缩包内所有文件的相对路径列表。"""
        return list(self.file_index.keys())
    
    def get_statistics(self) -> dict:
        """获取压缩包统计信息。"""
        total_files = len(self.file_index)
        total_size = sum(f.get('size', 0) for f in self.file_index.values())
        compressed_total = sum(f.get('compressed_size', 0) for f in self.file_index.values())

        return {
            'total_files': total_files,
            'total_uncompressed_size': total_size,
            'total_compressed_size': compressed_total,
            'package_size_bytes': len(self.data),
            'package_size_mb': len(self.data) / (1024 * 1024),
            'format': self.detected_format,
            'skipped_count': len(self.skipped),
            'skipped': list(self.skipped),
        }


def detect_and_parse_package(
    filepath: str = None,
    data: bytes = None,
    max_package_size: int = MAX_PACKAGE_SIZE,
    max_file_size: int = MAX_FILE_SIZE,
    origin_name: str = None,
) -> Optional[CompressionPackage]:
    """检测并解析压缩包。

    Args:
        filepath: 文件路径（可选）
        data: 压缩包二进制数据（优先使用）
        max_package_size: 最大包尺寸限制（字节）
        max_file_size: 最大单文件限制（字节）
        origin_name: 显式指定源名（用于扩展名检测；默认取 filepath）

    Returns:
        解析后的 CompressionPackage 实例（含 is_valid / error_reason / skipped），
        无法读取源数据时返回 None。
    """
    if data is None and filepath is not None:
        try:
            with open(filepath, 'rb') as f:
                data = f.read(max_package_size + 1)
        except OSError:
            return None

    if data is None:
        return None

    pkg = CompressionPackage.__new__(CompressionPackage)
    pkg.origin_name = origin_name or filepath or "unknown"
    pkg.data = data
    pkg.file_index = {}
    pkg.is_valid = False
    pkg.error_reason = None
    pkg.skipped = []
    pkg.detected_format = None
    pkg.MAX_PACKAGE_SIZE = max_package_size
    pkg.MAX_FILE_SIZE = max_file_size
    pkg._parse()
    return pkg


def is_compression_format(filename: str) -> bool:
    """检查文件名是否为支持的压缩包格式。"""
    lower = filename.lower()
    if lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz'):
        return True
    ext = os.path.splitext(lower)[1]
    return ext in ('.zip', '.jar', '.war')


def get_compression_format(filename: str) -> Optional[str]:
    """获取压缩包格式扩展名。"""
    lower = filename.lower()
    if lower.endswith('.tar.gz'):
        return '.tar.gz'
    if lower.endswith('.tgz'):
        return '.tgz'
    if lower.endswith('.tar'):
        return '.tar'
    ext = os.path.splitext(lower)[1]
    if ext in ('.zip', '.jar', '.war'):
        return ext
    return None


def collect_archive_index(pkg: CompressionPackage, archive_rel_path: str) -> List[Dict[str, Any]]:
    """生成压缩包内文件索引（相对路径、大小、mtime、格式）。"""
    entries = []
    for rel, info in pkg.get_file_index().items():
        entries.append({
            'archive': archive_rel_path,
            'inner_path': rel,
            'virtual_path': f"{archive_rel_path}!{rel}",
            'size': info.get('size', 0),
            'mtime': info.get('mtime'),
            'format': pkg.detected_format,
        })
    return entries


if __name__ == "__main__":
    # 简单测试
    import tempfile
    import zipfile
    
    # 创建一个测试 ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test.txt", "Hello World test content")
        zf.writestr("src/main.py", "def hello():\n    print('Hello')")
        zf.writestr("config.yaml", "key: value")
    
    data = buf.getvalue()
    pkg = detect_and_parse_package(data=data)
    
    if pkg:
        stats = pkg.get_statistics()
        print(f"包统计: {stats}")
        
        for fname in pkg.get_file_list():
            info = pkg.get_file(fname)
            if info:
                print(f"文件: {fname}, 大小: {info['size']}, 内容: {info['content'][:50]}...")