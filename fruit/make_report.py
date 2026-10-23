"""讀取真實訓練成果，產生中文期中專案報告 PDF，並同步更新根目錄 README 的結果表。"""
import json
from datetime import date
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from cnn import CNN

# (字型路徑, ttc 子字型編號)；reportlab 只吃 TrueType 外框，Noto CJK 這類 CFF 字型無法使用。
FONT_CANDIDATES = [
    ('C:/Windows/Fonts/msjh.ttc', 0),                                # Windows 微軟正黑體
    ('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', None),  # macOS
    ('/Library/Fonts/Arial Unicode.ttf', None),
    ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 0),            # Linux／Colab：apt install fonts-wqy-zenhei
]
CLASS_ZH = {'freshapples': '新鮮蘋果', 'freshbanana': '新鮮香蕉', 'freshoranges': '新鮮橘子',
            'rottenapples': '腐爛蘋果', 'rottenbanana': '腐爛香蕉', 'rottenoranges': '腐爛橘子'}
PAGE_WIDTH = 485  # A4 扣掉左右邊界後可用寬度


def register_zh_font():
    """依序嘗試系統中文字型；都不能用時改用 reportlab 內建的繁中 CID 字型，任何系統都能產生 PDF。"""
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont('ZH', path, **({} if index is None else {'subfontIndex': index})))
                return 'ZH'
            except Exception:
                continue
    pdfmetrics.registerFont(UnicodeCIDFont('MSung-Light'))
    return 'MSung-Light'


def model_rows(n_classes, size):
    """逐區塊跑一次假輸入，取得輸出尺寸與參數量。"""
    model = CNN(n_classes).eval()
    x = torch.zeros(1, 3, size, size)
    shape = lambda t: '×'.join(map(str, t.shape[1:]))
    rows = [['層', '輸出尺寸', '參數量'], ['輸入 RGB 影像', shape(x), '0']]
    feats = list(model.features)
    with torch.no_grad():
        for k in range(0, len(feats), 4):  # 每 4 層是一個 Conv-BN-ReLU-MaxPool 區塊
            block = torch.nn.Sequential(*feats[k:k+4]); x = block(x)
            conv = feats[k]
            rows.append([f'Conv 3×3（{conv.in_channels}→{conv.out_channels}）+ BatchNorm + ReLU + MaxPool',
                         shape(x), f'{sum(p.numel() for p in block.parameters()):,}'])
        for layer in model.classifier:
            x = layer(x)
            pool = getattr(layer, 'output_size', None)
            name = {'AdaptiveAvgPool2d': '自適應平均池化 ' + ('×'.join(map(str, pool)) if isinstance(pool, tuple) else str(pool)),
                    'Flatten': '攤平', 'Dropout': f'Dropout {getattr(layer, "p", "")}',
                    'Linear': f'全連接 {getattr(layer, "in_features", "")}→{getattr(layer, "out_features", "")}'}
            rows.append([name.get(type(layer).__name__, type(layer).__name__), shape(x),
                         f'{sum(p.numel() for p in layer.parameters()):,}'])
    total = sum(p.numel() for p in model.parameters())
    rows.append(['總計', '', f'{total:,}'])
    return rows, total


def confusion_analysis(r):
    classes = [v['class'] for v in r['per_class']]
    cm = np.asarray(r['confusion_matrix'])
    total, wrong = int(cm.sum()), int(cm.sum() - np.trace(cm))
    state = [c.startswith('fresh') for c in classes]
    fruit = [c.replace('fresh', '').replace('rotten', '') for c in classes]
    off = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j]]
    return {
        'total': total, 'wrong': wrong,
        'freshness_only': sum(n for n, i, j in off if fruit[i] == fruit[j]),
        'fruit_only': sum(n for n, i, j in off if state[i] == state[j]),
        'both': sum(n for n, i, j in off if fruit[i] != fruit[j] and state[i] != state[j]),
        'binary_acc': sum(int(cm[i, j]) for i in range(len(cm)) for j in range(len(cm)) if state[i] == state[j]) / total,
        'fruit_acc': sum(int(cm[i, j]) for i in range(len(cm)) for j in range(len(cm)) if fruit[i] == fruit[j]) / total,
        'top': [(n, classes[i], classes[j], n / int(cm[i].sum())) for n, i, j in sorted(off, reverse=True)[:3]],
    }


def update_readme(path, r, ca, params):
    """改寫 README 中 <!-- results:start --> 與 <!-- results:end --> 之間的結果表。"""
    start, end = '<!-- results:start -->', '<!-- results:end -->'
    if not path.exists():
        return False
    text = path.read_text(encoding='utf-8')
    if start not in text or end not in text:
        return False
    device = (r.get('environment') or {}).get('gpu') or r['device'].upper()
    block = '\n'.join([
        '| 指標 | 數值 |', '|---|---|',
        f'| Test Accuracy | {r["test_accuracy"]:.2%} |',
        f'| Macro F1 | {r["macro_f1"]:.4f} |',
        f'| 新鮮 / 腐爛 二分準確率 | {ca["binary_acc"]:.2%} |',
        f'| 多數類別基準 Accuracy | {r["majority_baseline_accuracy"]:.2%} |',
        f'| 模型參數量 | {params:,} |',
        f'| 訓練 | {r["epochs_run"]} 回合（最佳第 {r["best_epoch"]} 回合），{device} 約 {r["elapsed_seconds"]/60:.1f} 分鐘 |',
    ])
    new = text[:text.index(start)+len(start)] + '\n' + block + '\n' + text[text.index(end):]
    path.write_text(new, encoding='utf-8')
    return True


def main():
    root = Path(__file__).resolve().parent
    res = root/'result'
    r = json.loads((res/'metrics.json').read_text(encoding='utf-8'))
    eda = json.loads((res/'eda_stats.json').read_text(encoding='utf-8')) if (res/'eda_stats.json').exists() else None
    demo = json.loads((res/'demo_predictions.json').read_text(encoding='utf-8')) if (res/'demo_predictions.json').exists() else None
    size = r.get('image_size') or r['metadata']['image_size']
    arch_rows, params = model_rows(len(r['per_class']), size)
    ca = confusion_analysis(r)
    zh = lambda c: f'{CLASS_ZH.get(c, c)}（{c}）'

    font = register_zh_font()
    body = ParagraphStyle('body', fontName=font, fontSize=10, leading=17,
                          spaceAfter=9, wordWrap='CJK')
    title = ParagraphStyle('title', parent=body, fontSize=22, leading=30,
                           textColor=colors.HexColor('#155e63'), spaceAfter=18)
    heading = ParagraphStyle('heading', parent=body, fontSize=14, leading=22,
                             textColor=colors.HexColor('#155e63'), spaceBefore=12)
    story = []
    def p(text, style=body): story.append(Paragraph(text, style))
    def table(rows, widths):
        t = Table([[Paragraph(str(v), body) for v in row] for row in rows], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2f1')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#b7cecd')),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7)]))
        story.append(t)
    def img(path, width=PAGE_WIDTH, max_height=600):
        """依原圖比例縮放；檔案不存在時寫明要執行哪支程式。"""
        if not path.exists():
            p(f'<font color="#b83232">（尚未產生 {path.name}）</font>')
            return
        w, h = PILImage.open(path).size
        width = min(width, max_height * w / h)
        story.append(Image(str(path), width=width, height=width * h / w))

    # ---------- 封面與動機 ----------
    p('水果新鮮度 CNN 分類', title)
    p(f'機器學習期中專案報告 | {date.today().isoformat()}')
    p('一、題目與動機', heading)
    p('生鮮賣場、物流與家庭中，水果是否腐爛多半靠人工目視判斷，既耗時、標準也不一致。若能用一張照片自動判斷水果是新鮮還是腐爛，'
      '就能協助品管分揀、及早挑出壞果並減少食物浪費。本專案以卷積神經網路（CNN）從照片同時判斷水果種類（蘋果、香蕉、橘子）'
      '與外觀新鮮度，共六類。')
    p('選用 Kaggle「Fruits fresh and rotten for classification」資料集：公開、標註清楚、六個類別數量適中，'
      '而且是影像資料，正好練習課堂所學的 CNN。')
    p('二、如何符合公告', heading)
    p('依期中專案公告，完成可執行的個人影像分類專案，涵蓋資料處理、模型訓練及結果評估。資料改為 fruit.zip 中的水果影像，不沿用課堂 Fashion-MNIST 或溫度範例資料。')
    p('第 8 週報告時，現場以保存的最佳模型執行 predict.py 推論照片並說明程式碼；需要時也可用 train.py 重新訓練。')
    p('三、原註解與檔案保存方式', heading)
    p('fruit/original/ 完整保存課堂原始 cnn.py、dataset.py、test2.py，包含全部註解；fruit/previous/ 完整保存先前 LEGO 版本的三個程式與 archive.zip 資料。水果版在 fruit/ 建立獨立模組，避免原有雙 main、額外訓練迴圈及不同任務混用。')

    # ---------- 資料 ----------
    story.append(PageBreak())
    p('四、資料集與資料探索（EDA）', heading)
    m = r['metadata']
    p(f'壓縮檔有 dataset/ 與 dataset/dataset/ 兩套結構。本次僅採 dataset/train/ 與 dataset/test/，共 {m["canonical_entries"]:,} 張候選圖；依原始檔案 SHA-256 排除 {m["duplicates_removed"]:,} 張完全重複圖，跳過 {len(m["skipped"])} 張無法讀取圖，保留 {m["unique_images"]:,} 張。壓縮檔不解壓、不修改，RGB 縮圖載入記憶體供各回合使用。')
    rows = [['類別', '訓練', '驗證', '測試']]
    for c in m['counts']['train']:
        rows.append([zh(c)] + [m['counts'][s][c] for s in ('train', 'val', 'test')])
    rows.append(['總計'] + [sum(m['counts'][s].values()) for s in ('train', 'val', 'test')])
    table(rows, [208, 92, 92, 93])
    if eda:
        p(f'訓練集中最多的是{zh(eda["largest_class"])}、最少的是{zh(eda["smallest_class"])}，相差約 {eda["imbalance_ratio"]:.1f} 倍，'
          '有輕微類別不平衡，因此除了 accuracy 之外也以 Macro F1 評估。'
          f'原始圖片尺寸差異很大（抽樣 {eda["size_sample"]} 張：寬 {eda["width"][0]}~{eda["width"][2]} px、中位數 {eda["width"][1]:.0f} px；'
          f'高 {eda["height"][0]}~{eda["height"][2]} px），所以訓練前統一縮放到 {size}×{size}。')
    img(res/'eda_class_counts.png', 440, 190)
    img(res/'eda_sizes.png', 230)
    story.append(PageBreak())
    p('各類別範例', heading)
    img(res/'eda_samples.png', 420, 640)

    # ---------- 方法 ----------
    story.append(PageBreak())
    p('五、程式設計與理由', heading)
    table([['檔案', '修改與設計理由'],
        ['dataset.py', '由 ZIP 讀取六類影像；捨棄重複資料夾，移除完全相同內容。保留原 test，只從 train 逐類抽 15% 作驗證，種子 42。輸出 manifest 與稽核 JSON，讓每張圖的去向可追查。'],
        ['eda.py', '畫出各類別張數、每類範例原圖與原始尺寸分布，作為前處理決策的依據。'],
        ['cnn.py', f'影像輸入改為 RGB 三通道 {size}×{size}；三組 16/32/64 通道卷積搭配 BatchNorm、ReLU 與池化。用自適應池化取代固定 28×28 尺寸假設，Dropout 0.3 降低過擬合，輸出六類 logits。'],
        ['train.py', '單一入口完成讀取、訓練、驗證與最終測試。CrossEntropy 直接接 logits，Adam 學習率 0.001、weight decay 0.0001。僅依驗證 loss 保存最佳權重，連續五回合未改善即停止，避免測試集參與調參。'],
        ['評估輸出', '訓練／驗證 loss 與 accuracy 曲線、測試 accuracy、macro F1、逐類 precision/recall/F1、混淆矩陣、分錯圖片及逐張預測與信心度。另報告永遠猜訓練最多類別的基準，確認模型是否優於簡單猜測。'],
        ['predict.py', '可一次預測多張照片或整個資料夾，輸出 Demo 結果圖；讀取保存的類別名稱、尺寸與正規化設定，推論使用相同前處理。'],
        ['make_report.py', '讀取實際結果自動產生本報告，並同步更新 README 的結果表，避免數字與實測不一致。']], [92, 393])
    p('前處理與資料增強', heading)
    p(f'圖片校正 EXIF 方向、轉 RGB、縮至 {size}×{size}，像素轉 0~1 後以 (x-0.5)/0.5 映射到 -1~1。只在訓練時隨機水平翻轉，驗證及測試固定前處理；這裡使用固定尺度，沒有用全體影像估計統計量。固定亂數種子；CUDA 可用時自動選擇 GPU。')
    story.append(PageBreak())
    p('六、模型架構', heading)
    table(arch_rows, [255, 110, 120])
    p(f'整個模型只有 {params:,} 個參數，比課堂常見的全連接網路小很多：卷積層共用權重，再用自適應平均池化把特徵圖壓成 2×2，最後一層全連接只需 256×6 個權重。')

    # ---------- 結果 ----------
    story.append(PageBreak())
    p('七、訓練與結果評估', heading)
    env = r.get('environment') or {}
    env_txt = (f'環境：Python {env.get("python")}、PyTorch {env.get("torch")}、'
               f'{env.get("gpu") or ("CPU " + (env.get("cpu") or "")).strip()}。') if env else ''
    stop_txt = ('最佳權重出現在最後一回合，驗證 loss 仍在下降，代表增加訓練回合可能還會進步。'
                if r['best_epoch'] == r['epochs_run'] else
                f'第 {r["best_epoch"]} 回合後驗證 loss 不再下降，於第 {r["epochs_run"]} 回合提前停止。')
    p(f'實際執行 {r["epochs_run"]} 回合，最佳驗證權重在第 {r["best_epoch"]} 回合；裝置 {r["device"]}，包含資料讀取及訓練共 {r["elapsed_seconds"]/60:.1f} 分鐘。{stop_txt}{env_txt}')
    table([['指標', '測試結果'], ['Accuracy', f'{r["test_accuracy"]:.2%}'],
        ['Macro F1', f'{r["macro_f1"]:.4f}'], ['新鮮 / 腐爛 二分準確率', f'{ca["binary_acc"]:.2%}'],
        ['水果種類準確率（不看新鮮度）', f'{ca["fruit_acc"]:.2%}'],
        ['多數類別基準 Accuracy', f'{r["majority_baseline_accuracy"]:.2%}'],
        ['Cross entropy', f'{r["test_loss"]:.4f}']], [240, 245])
    story.append(Spacer(1, 10))
    table([['類別', 'Precision', 'Recall', 'F1', '樣本數']] +
          [[zh(v['class']), f'{v["precision"]:.3f}', f'{v["recall"]:.3f}', f'{v["f1"]:.3f}', v['support']] for v in r['per_class']], [185, 75, 75, 75, 75])
    story.append(Spacer(1, 14))
    img(res/'learning_curves.png')

    # ---------- 混淆分析 ----------
    story.append(PageBreak())
    p('八、混淆分析', heading)
    img(res/'confusion_matrix.png', 400)
    p(f'測試集 {ca["total"]:,} 張中分錯 {ca["wrong"]:,} 張（{ca["wrong"]/ca["total"]:.1%}）。其中 {ca["freshness_only"]} 張是水果種類對、新鮮度判錯；'
      f'{ca["fruit_only"]} 張是新鮮度對、水果種類判錯；{ca["both"]} 張兩者都錯。')
    if ca['top']:
        items = '<br/>'.join(f'{k}. {zh(t)} → {zh(pr)}：{n} 張（占該類 {share:.1%}）' for k, (n, t, pr, share) in enumerate(ca['top'], 1))
        p('最常見的三種錯誤：<br/>' + items)
    if ca['freshness_only'] >= max(ca['fruit_only'], ca['both']):
        p('錯誤主要集中在「同一種水果的新鮮度」：模型很少認錯水果種類，難的是分辨輕微腐爛。'
          f'在 {size}×{size} 的低解析度下，小斑點、陰影或反光都可能被誤認，這也是提高解析度最可能改善的部分。')
    else:
        p('錯誤主要來自水果種類互判，代表不同水果在縮小後的形狀與顏色特徵仍不夠可分，可考慮提高解析度或加強資料增強。')

    # ---------- 錯誤案例與 Demo ----------
    story.append(PageBreak())
    p('九、錯誤案例', heading)
    p('下圖為測試集中分錯的圖片，依模型信心度由高到低排列（最有把握卻答錯的排前面），標題為真實類別、預測類別與信心度。')
    img(res/'misclassified.png', PAGE_WIDTH, 560)
    story.append(PageBreak())
    p('十、Demo：自行拍攝的照片', heading)
    if demo:
        p(f'以 predict.py 對資料集以外的 {len(demo)} 張照片推論（綠色為預測新鮮、紅色為預測腐爛）：')
        img(res/'demo_predictions.png', PAGE_WIDTH, 420)
        table([['照片', '預測', '信心度', '第二名']] +
              [[d['file'], zh(d['top3'][0][0]), f'{d["top3"][0][1]:.1%}',
                f'{CLASS_ZH.get(d["top3"][1][0], d["top3"][1][0])} {d["top3"][1][1]:.1%}' if len(d['top3']) > 1 else '']
               for d in demo], [150, 150, 70, 115])
    else:
        p('<font color="#b83232">（尚未執行 Demo：python fruit/predict.py 照片資料夾）</font>')

    # ---------- 限制與結論 ----------
    story.append(PageBreak())
    p('十一、使用限制', heading)
    p('SHA-256 只能排除位元完全相同的影像，不能辨識旋轉、裁切或翻轉的近似圖；資料集離線增強來源與拍攝個體沒有可靠群組標籤，因此隨機驗證可能偏樂觀。保留原 test 也不等於保證拍攝來源獨立，建議以另外拍攝的照片測試（見 Demo）。')
    p('本模型只能在六個既定類別中選一個，未知水果、複雜背景、不同光源與不同相機都可能誤判；softmax 分數不是校準後的正確機率。分類代表外觀標籤，不能作為食品安全或可食用判斷。')
    p('十二、結論與未來改進', heading)
    concl = [f'以僅 {params:,} 個參數的小型 CNN、{size}×{size} 輸入，在 {ca["total"]:,} 張測試圖上達到 accuracy {r["test_accuracy"]:.2%}、Macro F1 {r["macro_f1"]:.4f}，'
             f'遠高於永遠猜最多類別的基準 {r["majority_baseline_accuracy"]:.2%}；只看新鮮 / 腐爛時準確率為 {ca["binary_acc"]:.2%}。']
    if ca['top']:
        n, t, pr, _ = ca['top'][0]
        concl.append(f'最主要的錯誤是把{zh(t)}判成{zh(pr)}（{n} 張），是下一步最值得改善的地方。')
    concl.append(stop_txt)
    p('<br/>'.join(f'• {c}' for c in concl))
    p('未來改進：<br/>'
      f'• 提高輸入解析度（例如 128×128），讓模型看得到細小的腐爛斑點。<br/>'
      '• 加入旋轉、縮放、亮度等資料增強，但避免改變色相，因為顏色是判斷新鮮度的關鍵。<br/>'
      '• 與 ImageNet 預訓練模型（如 ResNet18）的遷移學習比較準確率與參數量。<br/>'
      '• 收集更多不同背景、光線的真實照片測試，並以 Grad-CAM 檢查模型是否真的在看腐爛區域。')
    p('展示與重現', heading)
    p('安裝：python -m pip install -r fruit/requirements.txt<br/>訓練（含 EDA 與評估圖）：python fruit/train.py<br/>Demo：python fruit/predict.py 照片1.jpg 照片2.jpg（或照片資料夾）<br/>重建報告與 README 結果表：python fruit/make_report.py')
    p('輸出保存在 fruit/result/；PDF 保存在 fruit/output/pdf/。資料集 ZIP 檔案過大未上傳 GitHub，重新訓練前需放在 fruit/fruit.zip。')

    dest = root/'output/pdf/fruit_cnn_report.pdf'; dest.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont(font, 9); canvas.setFillColor(colors.HexColor('#526b6b'))
        canvas.drawString(42, 25, '水果新鮮度 CNN | 期中專案報告')
        canvas.drawRightString(553, 25, str(doc.page))
    SimpleDocTemplate(str(dest), pagesize=(595, 842), leftMargin=55, rightMargin=55,
        topMargin=42, bottomMargin=45).build(story, onFirstPage=footer, onLaterPages=footer)
    print(dest)
    if update_readme(root.parent/'README.md', r, ca, params):
        print('README 結果表已更新')


if __name__ == '__main__':
    main()
