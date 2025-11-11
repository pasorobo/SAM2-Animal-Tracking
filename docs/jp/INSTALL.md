# インストール

1. リポジトリをクローンしてサブモジュールを初期化
`git clone --recurse-submodules https://github.com/ecker-lab/SAM2-Animal-Tracking`
2. 環境のセットアップ（例: conda使用、python>=3.10）
`conda create --name sam2animal python=3.12`
`conda activate sam2animal`
3. PyTorchのインストール（torch>=2.5.1, torchvision>=0.20.1）
`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126`
4. リポジトリのインストール
`pip install -e .`
5. SAM2チェックポイントのダウンロード（sam2リポジトリの手順を参照）
```cd sam2/checkpoints && \
./download_ckpts.sh && \
cd ../..
```
6. （オプション）別のMMDetection環境のセットアップ
これは、MMDetection検出器を使用する特定の理由がある場合にのみ推奨されます。
`conda create --name mm_detect python=3.12`
`conda activate mm_detect`
その後、[MMDetectionのインストール手順](https://mmdetection.readthedocs.io/en/latest/get_started.html)に従ってください。
