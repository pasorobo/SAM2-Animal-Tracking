# 検出器ドキュメント

以下の検出器が標準でサポートされています。さらに多くの/カスタム検出器を簡単に追加できます。

- Huggingface
  - 例: [OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2), [Grounding Dino](https://huggingface.co/docs/transformers/model_doc/grounding-dino), [LLMDet](https://huggingface.co/fushh7/LLMDet)
- [mmDetection](https://github.com/open-mmlab/mmdetection)
  - 例: [Grounding Dino](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)
- 既存の検出結果

## Huggingface検出器

Huggingfaceの検出器を使用するのは簡単で、追加の手順は必要ありません。以下の引数を設定する必要があります：

- `--DETECTOR.MODEL_SRC huggingface`
- `--DETECTOR.MODEL <モデルID>`

例えば、LLMDet largeは以下のhuggingfaceモデルIDを持っています：`iSEE-Laboratory/llmdet_large`。追加の変更は必要ありません。

## MMDetection

MMDetectionモデルを使用するのはかなり複雑で、推奨されません。[公式ドキュメント](https://mmdetection.readthedocs.io/en/latest/get_started.html)に従って、`mmengine, mmcv, mmdetection`を使用して新しいPython環境をセットアップする必要があります。次に、モデル設定とチェックポイントをダウンロードして、このリポジトリの`mmcv`フォルダに配置する必要があります。その後、以下の引数を設定する必要があります：

- `--DETECTOR.CHECKPOINT_PATH <チェックポイントファイルへのパス>`
- `--DETECTOR.CONFIG_PATH <設定ファイルへのパス>`

**警告:** 検出と追跡には異なる環境が使用されます。すべての検出を事前計算し、標準環境を有効化してから追跡を開始する必要があります。

## 既存の検出結果の読み込み

既存の検出結果は、以下の引数を設定することで読み込むことができます。

- `--DETECTOR.LOAD_DETS True`
- `--DETECTOR.MODEL <モデル名>`

検出結果は以下のように配置する必要があります。

```
outputs/
├── <データセット名>/
│   ├── <データセット分割>/
│   │   ├── <モデル名>
│   │   │   ├── seq1.pt/
│   │   │   ├── seq2.jpg
│   │   │   └── seq3.jpg
```

検出結果は、テンソルの辞書のリストであることが期待されます。各リスト要素には、それぞれのフレームの検出が含まれています。辞書には`boxes`、`scores`、`labels`のキーがあります。`boxes`はtlwh形式の絶対ピクセル座標で、形状は`[N,4]`です。`scores`は0から1の間の信頼度スコアで、形状`[N]`のテンソルです。`labels`は整数で、形状`[N]`のテンソルです。
