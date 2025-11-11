# リアルタイムストリーミング追跡

このドキュメントでは、USBカメラまたは動画ファイルからオブジェクトを追跡するためのリアルタイムストリーミングモードの使用方法について説明します。

## 概要

ストリーミングモードは、ビデオフレームをリアルタイムで処理し、以下をサポートします：
- **USBウェブカメラ**（またはあらゆるカメラデバイス）
- **動画ファイル**（MP4、AVIなど）
- **複数オブジェクト追跡**と自動ID割り当て
- **適応的検出閾値**（オンラインOtsu法を使用）
- **リアルタイム可視化**（マスク、ボックス、IDを表示）
- **ビデオ出力**（結果の保存）

## クイックスタート

### 1. USBカメラ追跡

ウェブカメラ（カメラ0）からオブジェクトを追跡：

```bash
python run_streaming.py --source 0 --text-prompt animal
```

カメラ1、2などの場合：

```bash
python run_streaming.py --source 1 --text-prompt person
```

### 2. 動画ファイル追跡

動画ファイル内のオブジェクトを追跡：

```bash
python run_streaming.py --source /path/to/video.mp4 --text-prompt animal
```

### 3. 出力ビデオの保存

追跡結果をビデオファイルに保存：

```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --output-video outputs/tracking_result.mp4
```

## コマンドライン引数

### 入力設定

- `--source`: 入力ソース
  - カメラインデックス：`0`、`1`、`2`、...
  - 動画ファイル：`/path/to/video.mp4`
  - デフォルト：`0`（最初のカメラ）

- `--target-fps`: 処理のターゲットFPS
  - 処理速度を制限（低速なハードウェアに有用）
  - デフォルト：`None`（すべてのフレームを処理）
  - 例：`--target-fps 15`

### 検出設定

- `--text-prompt`: 検出用の事前定義テキストプロンプト
  - `person`: 人物を検出
  - `animal`: 動物を検出（犬、猫、鳥、馬、羊、牛）
  - `chimpanzee`: 類人猿/チンパンジーを検出
  - `bird`: 鳥を検出
  - `vehicle`: 車両を検出
  - `all`: すべてのオブジェクトを検出
  - `custom`: カスタムプロンプトを使用（`--custom-prompt`が必要）
  - デフォルト：`animal`

- `--custom-prompt`: カスタム検出プロンプト（カンマ区切り）
  - 例：`--text-prompt custom --custom-prompt "dog,cat,rabbit"`

- `--detector-model`: HuggingFace検出器モデル
  - デフォルト：`iSEE-Laboratory/llmdet_large`
  - 代替：`google/owlv2-base-patch16-ensemble`、`IDEA-Research/grounding-dino-base`

- `--detection-threshold`: 初期検出閾値（0-1）
  - デフォルト：`0.3`
  - 低い = より多くの検出（偽陽性が増える）
  - 高い = より少ない検出（オブジェクトを見逃す可能性）

- `--high-conf-threshold`: 新しいオブジェクトを追加するための高信頼度閾値
  - デフォルト：`0.4`

- `--use-adaptive-threshold`: 適応的閾値を有効化（Otsu法）
  - デフォルト：`True`
  - 検出スコア分布に基づいて閾値を自動調整

### SAM2設定

- `--sam2-checkpoint`: SAM2チェックポイントパス
  - デフォルト：`sam2/checkpoints/sam2.1_hiera_large.pt`
  - 利用可能：`sam2.1_hiera_large.pt`、`sam2.1_hiera_base_plus.pt`、`sam2.1_hiera_small.pt`、`sam2.1_hiera_tiny.pt`
  - 小さいモデル = 高速だが精度は低い

- `--sam2-config`: SAM2設定ファイル
  - デフォルト：`configs/sam2.1/sam2.1_hiera_l.yaml`
  - チェックポイントサイズと一致する必要がある

- `--sam2-device`: SAM2用デバイス
  - デフォルト：`cuda`
  - GPUが利用できない場合は`cpu`を使用（大幅に遅くなる）

- `--compile-sam2`: より高速な推論のためにSAM2エンコーダをコンパイル
  - デフォルト：`True`
  - PyTorch 2.0+が必要

### 可視化設定

- `--show-masks`: セグメンテーションマスクを表示
  - デフォルト：`True`

- `--show-boxes`: バウンディングボックスを表示
  - デフォルト：`True`

- `--show-ids`: トラックIDを表示
  - デフォルト：`True`

- `--show-scores`: 検出スコアを表示
  - デフォルト：`False`

- `--show-fps`: FPSカウンターを表示
  - デフォルト：`True`

- `--mask-alpha`: マスクの透明度（0-1）
  - デフォルト：`0.5`
  - 低い = より透明、高い = より不透明

### 出力設定

- `--output-video`: 出力ビデオファイルパス
  - デフォルト：`None`（出力なし）
  - 例：`outputs/result.mp4`

- `--output-fps`: 出力ビデオFPS
  - デフォルト：`30`

## キーボードコントロール

実行中：
- **`q`**: アプリケーションを終了
- **`p`**: 追跡を一時停止/再開
- **`r`**: トラッカーをリセット（すべてのトラックをクリアして再初期化）

## 使用例

### 例1: ウェブカメラから動物を追跡

```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --show-masks True \
  --show-ids True \
  --output-video outputs/webcam_tracking.mp4
```

### 例2: 動画ファイル内の人物を追跡

```bash
python run_streaming.py \
  --source videos/crowd.mp4 \
  --text-prompt person \
  --detection-threshold 0.4 \
  --output-video outputs/crowd_tracking.mp4
```

### 例3: カスタム設定で鳥を追跡

```bash
python run_streaming.py \
  --source videos/birds.mp4 \
  --text-prompt bird \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --mask-alpha 0.7 \
  --output-video outputs/birds_tracking.mp4
```

### 例4: 複数の動物種を追跡

```bash
python run_streaming.py \
  --source 1 \
  --text-prompt custom \
  --custom-prompt "dog,cat,rabbit,hamster" \
  --detection-threshold 0.35 \
  --show-fps True
```

### 例5: CPUのみモード（GPUなし）

```bash
python run_streaming.py \
  --source 0 \
  --text-prompt person \
  --sam2-device cpu \
  --detector-device cpu \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --target-fps 5
```

### 例6: 高性能追跡

```bash
python run_streaming.py \
  --source videos/test.mp4 \
  --text-prompt animal \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --compile-sam2 True \
  --use-adaptive-threshold True \
  --output-video outputs/high_perf.mp4
```

## VOS最適化（上級者向け - 2倍速ブースト！）

### VOS最適化とは？

VOS（ビデオオブジェクトセグメンテーション）最適化は、高度なPyTorchコンパイル技術を使用して**約2倍の速度向上**を実現します。[Gy920のsegment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)実装に基づいています。

**主な機能：**
- ✅ すべてのSAM2モジュールに`torch.compile`を適用（メモリエンコーダ、アテンション、プロンプトエンコーダ、マスクデコーダ）
- ✅ `num_maskmem`制限によるメモリ管理（メモリ爆発を防止）
- ✅ 約2倍高速な推論（初期ウォームアップ後）
- ⚠️ CUDA（GPU）が必要
- ⚠️ 最初の数フレームは遅い（コンパイルオーバーヘッド）

### VOS最適化を有効化

コマンドに`--vos-optimize True`を追加：

```bash
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --vos-optimize True
```

**期待されるパフォーマンス（RTX 3090）：**

| 構成 | FPS（標準） | FPS（VOS最適化） | 高速化 |
|---------------|----------------|---------------------|---------:|
| `hiera_large` | ~8 FPS | ~16 FPS | 2.0x |
| `hiera_base_plus` | ~12 FPS | ~24 FPS | 2.0x |
| `hiera_small` | ~20 FPS | ~40 FPS | 2.0x |

### VOSオプション

**`--vos-optimize`**: VOS最適化を有効化（デフォルト：`False`）
```bash
--vos-optimize True  # 2倍速ブーストを有効化
```

**`--num-maskmem`**: メモリに保持するフレーム数（デフォルト：`7`）
```bash
--num-maskmem 7  # 最後の7フレームを保持（メモリ爆発を防止）
```

**`--compile-mode`**: torch.compile最適化モード（デフォルト：`max-autotune`）
```bash
--compile-mode max-autotune     # 最速（起動が遅い）
--compile-mode reduce-overhead  # バランス型
--compile-mode default          # 保守的
```

### 例: 高性能追跡

```bash
# 小モデルでのVOS最適化追跡 = RTX 3090で約40 FPS
python run_streaming.py \
  --source 0 \
  --text-prompt animal \
  --vos-optimize True \
  --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --num-maskmem 7 \
  --compile-mode max-autotune \
  --output-video outputs/vos_tracking.mp4
```

### VOSデモスクリプト

簡略化されたデモが利用可能：

```bash
python examples/streaming_demo_vos.py --source 0 --prompt animal
```

### 重要な注意事項

1. **ウォームアップ期間**: 最初の5-10フレームは遅くなります（torch.compileオーバーヘッド）
2. **CUDAが必要**: VOS最適化にはGPUが必要です。CPUでは自動的に無効化されます。
3. **メモリ安定性**: `num_maskmem`は長時間セッション中のメモリ爆発を防ぎます
4. **コンパイル時間**: 初期起動時にコンパイルに約30-60秒かかります
5. **モデル互換性**: すべてのSAM2モデルサイズで動作します

### VOS最適化を使用するタイミング

**✅ 以下の場合にVOSを使用：**
- CUDA対応GPUがある
- リアルタイムパフォーマンスが重要（20 FPS以上が必要）
- 長時間セッション（数時間/数日）
- 高解像度ビデオの処理

**❌ 以下の場合はVOSを使用しない：**
- CPUのみで実行
- クイックテスト（ウォームアップオーバーヘッドが価値に見合わない）
- デバッグ（コンパイルによりデバッグが困難）
- GPUメモリが限られている（8GB未満）

## パフォーマンス最適化のヒント

### 1. より小さいSAM2モデルを使用

リアルタイムパフォーマンスには、より小さいSAM2モデルを使用：

| モデル | 速度 | 精度 |
|-------|-------|----------|
| `sam2.1_hiera_tiny.pt` | 最速 | 低い |
| `sam2.1_hiera_small.pt` | 高速 | 良い |
| `sam2.1_hiera_base_plus.pt` | 中速 | より良い |
| `sam2.1_hiera_large.pt` | 低速 | 最良 |

```bash
# 高速追跡
--sam2-checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml
```

### 2. モデルコンパイルを有効化

PyTorch 2.0+のモデルコンパイルにより速度が大幅に向上：

```bash
--compile-sam2 True
```

### 3. ターゲットFPSを制限

1秒あたりのフレーム処理数を減らす：

```bash
--target-fps 15  # 15 FPSのみ処理
```

### 4. GPUを使用

GPUが利用可能な場合は常に使用：

```bash
--sam2-device cuda --detector-device cuda
```

### 5. 検出閾値を調整

高い閾値 = より少ない検出 = より高速な処理：

```bash
--detection-threshold 0.5  # 高信頼度の検出のみ
```

### 6. 不要な可視化を無効化

```bash
--show-scores False  # スコア表示を無効化
```

## アーキテクチャの詳細

### オンライン処理パイプライン

```
フレーム入力 → 検出 → SAM2伝播 → 追跡ロジック → 可視化
     ↓         ↓          ↓            ↓           ↓
  カメラ/    ゼロショット  セグメン    追加/削除    リアルタイム
   動画      検出        テーション   オブジェクト   表示
                        マスク
```

### バッチモードとの主な違い

| 機能 | バッチモード（run.py） | ストリーミングモード（run_streaming.py） |
|---------|---------------------|--------------------------------------|
| 入力 | 画像フォルダ | カメラ/ビデオストリーム |
| 処理 | オフライン（全フレーム） | オンライン（フレームごと） |
| 検出 | 事前計算 | リアルタイム |
| 閾値 | Otsu（グローバル） | Otsu（オンライン、適応的） |
| レイテンシ | N/A | 低（リアルタイム） |
| 使用例 | 評価 | ライブアプリケーション |

### 適応的閾値

ストリーミングモードは、適応的検出閾値のために**オンラインOtsu法**を使用：

1. 最後の100個の検出スコアの履歴を維持
2. 毎フレームOtsu閾値を再計算
3. 変化するシーン条件に適応
4. 手動の閾値調整が不要

## トラブルシューティング

### 問題: 低FPS

**解決策：**
- より小さいSAM2モデルを使用（`sam2.1_hiera_small.pt`または`sam2.1_hiera_tiny.pt`）
- `--compile-sam2 True`を有効化
- `--target-fps`を削減
- オブジェクト数を減らすために`--detection-threshold`を増加
- GPUを使用（`--sam2-device cuda`）

### 問題: カメラが見つからない

**解決策：**
- カメラインデックスを確認（`--source 0`、`--source 1`などを試す）
- カメラが接続されているか確認：`ls /dev/video*`（Linux）
- カメラドライバをインストール
- 異なるソースインデックスを試す

### 問題: 偽陽性が多すぎる

**解決策：**
- `--detection-threshold`を増加（例：`0.4`または`0.5`）
- `--high-conf-threshold`を増加
- より具体的なテキストプロンプトを使用（例：`--text-prompt custom --custom-prompt "golden retriever"`）

### 問題: オブジェクトの見逃し

**解決策：**
- `--detection-threshold`を減少（例：`0.2`または`0.25`）
- より広範なテキストプロンプトを使用（例：`bird`の代わりに`--text-prompt animal`）
- 適応的閾値を無効化：`--use-adaptive-threshold False`

### 問題: CUDAメモリ不足

**解決策：**
- より小さいSAM2モデルを使用
- より少ないFPSを処理：`--target-fps 10`
- CPUモードを使用：`--sam2-device cpu --detector-device cpu`

## 技術的詳細

### メモリ管理

- 最近のフレーム履歴のみを維持（設定可能）
- 古い追跡データを自動的にクリーンアップ
- SAM2フレームストレージに一時ディレクトリを使用

### スレッドモデル

現在はシングルスレッド。今後の改善：
- 検出と追跡の個別スレッド
- 非同期可視化
- フレームバッファキュー

### レイテンシ

フレームあたりの典型的なレイテンシ：
- 検出：20-50ms（検出器モデルに依存）
- SAM2伝播：30-100ms（SAM2モデルとオブジェクト数に依存）
- 可視化：5-10ms
- **合計: 約60-160ms（6-17 FPS）** RTX 3090で

より良いリアルタイムパフォーマンスのために：
- `sam2.1_hiera_tiny.pt`を使用：約30 FPS
- `sam2.1_hiera_small.pt`を使用：約20 FPS
- `sam2.1_hiera_large.pt`を使用：約8 FPS

## 制限事項

1. **ハードウェア要件**: リアルタイムパフォーマンスにはGPUが必要
2. **シングルスレッド**: まだ並列処理なし
3. **メモリ**: 一時フレームファイルを蓄積
4. **オブジェクトの永続性**: 長時間失われたオブジェクトは新しいIDを再割り当てされる可能性

## 今後の拡張

- [ ] マルチスレッド処理
- [ ] フレームキューバッファリング
- [ ] リモートストリーミング用のWebRTCサポート
- [ ] 統合用のREST API
- [ ] モバイルデバイスサポート
- [ ] 複数カメラサポート

## サポート

問題や質問については：
- GitHub Issues: [課題を作成](https://github.com/ecker-lab/SAM2-Animal-Tracking/issues)
- 既存のドキュメントを確認：`docs/TRACKING.md`、`docs/DETECTION.md`
