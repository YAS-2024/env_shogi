# 使用するNVIDIAのCUDA Dockerイメージ 
FROM python:3.9-slim

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

# OSパッケージのインストール
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    curl \
    unzip \
    file \
    xz-utils \
    sudo \
    python3 \
    python3-pip \
    lsb-release \
    gnupg \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 日本語フォントのインストール
ENV NOTO_DIR /usr/share/fonts/opentype/notosans
RUN mkdir -p ${NOTO_DIR} && \
    wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip && \
    unzip ./noto.zip -d ${NOTO_DIR}/ && \
    chmod a+r ${NOTO_DIR}/NotoSans* && \
    rm ./noto.zip

# ワーキングディレクトリの設定
WORKDIR /workspace

# コンテナ起動時のデフォルトコマンド
CMD ["bash"]
