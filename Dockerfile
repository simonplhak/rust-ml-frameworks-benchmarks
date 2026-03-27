FROM rust:latest AS builder
LABEL stage=builder-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    build-essential \
    cmake \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip && \
    unzip -q libtorch-shared-with-deps-2.9.0+cpu.zip && \
    rm libtorch-shared-with-deps-2.9.0+cpu.zip
ENV LIBTORCH=/app/libtorch
ENV LIBTORCH_INCLUDE=/app/libtorch
ENV LIBTORCH_LIB=/app/libtorch


COPY Cargo.* ./
COPY burn-example/Cargo.* burn-example/
COPY candle-example/Cargo.*  candle-example/
COPY tch-example/Cargo.* tch-example/
COPY common/Cargo.* common/
RUN mkdir -p burn-example/src \
    && touch burn-example/src/lib.rs \
    && mkdir burn-example/benches \
    && touch burn-example/benches/model.rs
RUN mkdir -p candle-example/src \
    && touch candle-example/src/lib.rs \
    && mkdir candle-example/benches \
    && touch candle-example/benches/model.rs
RUN mkdir -p tch-example/src \
    && touch tch-example/src/lib.rs \
    && mkdir tch-example/benches \
    && touch tch-example/benches/model.rs
RUN mkdir -p common/src \
    && touch common/src/lib.rs

COPY pyproject.toml uv.lock .python-version README.md ./
ENV PATH="/root/.local/bin:${PATH}"
RUN uv sync

RUN cargo build --release --workspace
RUN cargo clean -p common -p burn-example -p candle-example -p tch-example -r

RUN rm burn-example/src/lib.rs candle-example/src/lib.rs tch-example/src/lib.rs common/src/lib.rs

COPY ./burn-example/ ./burn-example
COPY ./candle-example ./candle-example
COPY ./tch-example ./tch-example
COPY ./common ./common
COPY ./run_benchmarks.py .

# todo: move to other env variables
ENV LD_LIBRARY_PATH=/app/libtorch/lib

RUN cargo build --release --workspace

ENTRYPOINT ["uv", "run", "/app/run_benchmarks.py"]
