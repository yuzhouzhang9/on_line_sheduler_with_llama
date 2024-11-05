import datetime
import os
import time
from functools import wraps
from contextlib import contextmanager

class Logger:
    log_file_path = "log/application.log"

    @classmethod
    def set_log_file_path(cls, path=str(time.time())):
        path = "log/" + path                                        
        print(f"Creating log file at {path}... Done\n")
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

    @classmethod
    def measure_performance(cls, func, log_file='performance.log'):
        """
        装饰器函数，用于衡量函数的执行时间
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            message = f"函数 {func.__name__} 执行时间: {execution_time:.9f} 秒"
            cls.log(message)
            return result, execution_time
        return wrapper

    @classmethod
    @contextmanager
    def measure_block_time(cls, block_name="代码块"):
        """
        上下文管理器，用于衡量代码块的执行时间
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            message = f"{block_name} 执行时间: {execution_time:.9f} 秒"
            cls.log(message)

# 测试使用
if __name__ == "__main__":
    Logger.set_log_file_path("test_log.log")

    # 使用函数装饰器衡量函数执行时间
    @Logger.measure_performance
    def example_function():
        time.sleep(1)  # 模拟耗时操作
        return "Function Complete"

    result, execution_time = example_function()
    print(f"Result: {result}, Execution Time: {execution_time:.9f} 秒")

    # 使用上下文管理器衡量代码块执行时间
    with Logger.measure_block_time("测试代码块"):
        time.sleep(2)  # 模拟耗时代码块
