apt-get update && apt-get install -y musl-tools
wget https://github.com/facebook/zstd/releases/download/v1.5.5/zstd-1.5.5.tar.gz
tar -xzvf zstd-1.5.5.tar.gz
cd zstd-1.5.5
make CC="musl-gcc -static" ZSTD_LIB_COMPRESSION=0 ZSTD_LIB_DECOMPRESSION=1 zstd-release
