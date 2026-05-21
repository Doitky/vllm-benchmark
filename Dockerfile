FROM python:3.11-slim

WORKDIR /app

# 替换 apt 源为阿里云镜像
RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY vllm_benchmark.py /app/
COPY run_benchmarks.py /app/
#COPY LICENSE /app/

# 禁用 pip 进度条（避免线程问题）
ENV PIP_DISABLE_PROGRESS_BAR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 设置 pip 镜像为国内源
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple openai numpy

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run_benchmarks.py"]
CMD ["--help"]
