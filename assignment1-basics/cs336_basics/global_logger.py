import logging
from datetime import datetime
import os

class ColoredFormatter(logging.Formatter):
    """自定义带颜色的日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[41m',   # 红底白字
        'RESET': '\033[0m'        # 重置颜色
    }
    
    def format(self, record):
        # 获取原始格式的日志
        log_message = super().format(record)
        
        # 为控制台添加颜色
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            log_message = f"{color}{log_message}{reset}"
        
        return log_message

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建日志目录
log_directory = "log"
os.makedirs(log_directory, exist_ok=True)
current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file_name = f"{current_date}.log"

# 控制台处理器 - 带颜色
console_handler = logging.StreamHandler()
console_formatter = ColoredFormatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.DEBUG)

# 文件处理器 - 无颜色
file_handler = logging.FileHandler(os.path.join(log_directory, log_file_name))
file_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)