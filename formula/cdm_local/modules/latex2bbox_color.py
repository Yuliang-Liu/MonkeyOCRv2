import os
import re
import cv2
import json
import shutil
import logging
import subprocess
import signal
import numpy as np

from threading import Timer
from PIL import Image, ImageDraw
from modules.latex_processor import (
    normalize_latex,
    token_add_color_RGB,
    clean_latex
)
from modules.tokenize_latex.tokenize_latex import tokenize_latex


tabular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amssymb}
\usepackage{upgreek}
\usepackage{amsmath}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

formular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{amssymb}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

formular_template_zh = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{CJK}
\usepackage{amssymb}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{CJK}{UTF8}{gkai}
\begin{displaymath}
%s
\end{displaymath}
\end{CJK}
\end{document}
"""


def run_cmd(cmd, timeout_sec=30):
    timeout_sec = int(os.environ.get("CDM_PDFLATEX_TIMEOUT", timeout_sec))
    proc = subprocess.Popen(
        cmd,
        shell=True,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def kill_proc(p):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()
        
def convert_pdf2img(pdf_filename, png_filename):
    png_root = os.path.splitext(png_filename)[0]
    if shutil.which("pdftocairo"):
        cmd = "pdftocairo -png -singlefile -r 200 %s %s" % (pdf_filename, png_root)
    elif shutil.which("pdftoppm"):
        cmd = "pdftoppm -png -singlefile -r 200 %s %s" % (pdf_filename, png_root)
    elif shutil.which("magick"):
        cmd = "magick -density 200 -quality 100 %s %s" % (pdf_filename, png_filename)
    else:
        cmd = "convert -density 200 -quality 100 %s %s" % (pdf_filename, png_filename)
    os.system(cmd)

def crop_image(image_path, pad=8):
    img = Image.open(image_path).convert("L")
    img_data = np.asarray(img, dtype=np.uint8)
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        y_min = 0
        y_max = 10
        x_min = 0
        x_max = 10
    else:
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        
    img = Image.open(image_path).convert("RGB").crop((x_min-pad, y_min-pad, x_max+pad, y_max+pad))
    img.save(image_path)

def extrac_bbox_from_color_image(image_path, color_list):
    img = cv2.imread(image_path)
    bbox_list = []
    for target_color in color_list:
        r, g, b = target_color
        target_rgb = np.array([b, g, r], dtype=np.uint8)
        mask = np.all(img == target_rgb, axis=2)
        coords = np.argwhere(mask)
        if coords.size > 0:
            x_min, y_min = coords[:, 1].min(), coords[:, 0].min()
            x_max, y_max = coords[:, 1].max(), coords[:, 0].max()
            bbox_list.append([int(x_min-1), int(y_min-1), int(x_max+1), int(y_max+1)])
        else:
            bbox_list.append([])

    img = Image.open(image_path).convert("RGB").convert("L")
    img_bw = img.point(lambda x: 255 if x == 255 else 0, '1')
    img_bw.convert("RGB").save(image_path)
    return bbox_list

def contains_chinese(text):
    # 匹配中文字符的正则表达式范围
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def wrap_chinese_in_text(latex_text):
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]'
    # 匹配连续的中文字符
    chinese_sequence_pattern = chinese_pattern + '+'
    def replace_chinese(match):
        chinese_text = match.group(0)
        # 检查是否已经被\text{}包裹
        start_pos = match.start()
        end_pos = match.end()
        # 检查匹配位置前后是否有\text{和}
        before_text = latex_text[max(0, start_pos-6):start_pos]
        after_text = latex_text[end_pos:min(len(latex_text), end_pos+1)]
        if before_text.endswith('\\text{') and after_text.startswith('}'):
            return chinese_text
        else:
            return f'\\text{{{chinese_text}}}'
    # 替换所有连续的中文字符
    result = re.sub(chinese_sequence_pattern, replace_chinese, latex_text)
    return result
    
def latex2bbox_color(input_arg):
    latex, basename, output_path, temp_dir, total_color_list = input_arg
    save_token_vis = os.environ.get("CDM_SAVE_TOKEN_VIS", "1") != "0"
    if "tabular" in latex:
        template = tabular_template
    else:
        if contains_chinese(latex):
            template = formular_template_zh
            latex = latex.replace("，", ", ").replace("：", ": ").replace("；", "; ")
            latex = wrap_chinese_in_text(latex)
        else:
            template = formular_template
    output_bbox_path = os.path.join(output_path, 'bbox', basename+'.jsonl')
    output_vis_path = os.path.join(output_path, 'vis', basename+'.png')
    output_base_path = os.path.join(output_path, 'vis', basename+'_base.png')
    
    has_required_outputs = os.path.exists(output_bbox_path) and os.path.exists(output_base_path)
    has_optional_outputs = os.path.exists(output_vis_path) or not save_token_vis
    if has_required_outputs and has_optional_outputs:
        return
    
    try:
        latex = latex.replace("\n", " ")
        latex = latex.replace(r"\%", "<PERCENTAGETOKEN>")
        ret, new_latex = tokenize_latex(latex, middle_file=os.path.join(temp_dir, basename+'.txt'))
        if not(ret and new_latex):
            log = f"ERROR, Tokenize latex failed: {basename}."
            logging.info(log)
            new_latex = latex
        if contains_chinese(new_latex):
            new_latex = new_latex.replace("\\mathrm", "\\text")
        new_latex = new_latex.replace("< P E R C E N T A G E T O K E N >", r"\%")
        latex = normalize_latex(new_latex)
        token_list = []
        l_split = latex.strip().split(' ')
        color_list = total_color_list[0:len(l_split)]
        idx = 0
        while idx < len(l_split):
            l_split, idx, token_list = token_add_color_RGB(l_split, idx, token_list)

        rgb_latex = " ".join(l_split)
        for idx, color in enumerate(color_list):
            R, G, B = color
            rgb_latex = rgb_latex.replace(f"<color_{idx}>", f"{R},{G},{B}")

        if len(token_list) > 1300:
            paper_size = 3
        elif len(token_list) > 600:
            paper_size = 4
        else:
            paper_size = 5
        final_latex = template.replace("<PaperSize>", str(paper_size)) % rgb_latex
        
    except Exception as e:
        log = f"ERROR, Preprocess latex failed: {basename}; {e}."
        logging.info(log)
        return
    
    pre_name = output_path.replace('/', '_').replace('.','_') + '_' + basename
    tex_filename = os.path.join(temp_dir, pre_name+'.tex')
    log_filename = os.path.join(temp_dir, pre_name+'.log')
    aux_filename = os.path.join(temp_dir, pre_name+'.aux')
    
    with open(tex_filename, "w") as w: 
        print(final_latex, file=w)
    run_cmd(f"pdflatex -interaction=nonstopmode -output-directory={temp_dir} {tex_filename} >/dev/null")
    try:
        os.remove(tex_filename)
        os.remove(log_filename)
        os.remove(aux_filename)
    except:
        pass
    pdf_filename = tex_filename[:-4]+'.pdf'
    if not os.path.exists(pdf_filename):
        log = f"ERROR, Compile pdf failed: {pdf_filename}"
        logging.info(log)
    else:
        convert_pdf2img(pdf_filename, output_base_path)
        os.remove(pdf_filename)
        
        crop_image(output_base_path)
        bbox_list = extrac_bbox_from_color_image(output_base_path, color_list)
        vis = Image.open(output_base_path)
        draw = ImageDraw.Draw(vis)

        with open(output_bbox_path, 'w', encoding='utf-8') as f:
            for token, box in zip(token_list, bbox_list):
                item = {
                    "bbox": box,
                    "token": token
                }
                f.write(json.dumps(item, ensure_ascii=False)+'\n')

                if not box:
                    continue
                x_min, y_min, x_max, y_max = box
                draw.rectangle([x_min, y_min, x_max, y_max], fill=None, outline=(0,250,0), width=1)
                try:
                    draw.text((x_min, y_min), token, (250,0,0))
                except:
                    pass
            
            if save_token_vis:
                vis.save(output_vis_path)
