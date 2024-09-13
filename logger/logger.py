import datetime
import os
import time
class Logger:
    log_file_path = "log/application.log"

    @classmethod
    def set_log_file_path(cls, path = str(time.time())):
        path = "log/" + path
        cls.log_file_path = path
        cls._ensure_log_file_exists()

    @classmethod
    def _ensure_log_file_exists(cls):
        if not os.path.exists(cls.log_file_path):
            directory = os.path.dirname(cls.log_file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(cls.log_file_path, 'w') as f:
                f.write(f"Log file created at {datetime.datetime.now()}\n")

    @classmethod
    def log(cls, message):
        cls._ensure_log_file_exists()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} - {message}"
        
        try:
            with open(cls.log_file_path, "a") as log_file:
                log_file.write(log_entry + "\n")
        except Exception as e:
            print(f"无法写入日志文件: {str(e)}")

    @classmethod
    def log_error(cls, error_message):
        cls.log(f"错误: {error_message}")

    @classmethod
    def log_warning(cls, warning_message):
        cls.log(f"警告: {warning_message}")

    @classmethod
    def log_info(cls, info_message):
        cls.log(f"信息: {info_message}")