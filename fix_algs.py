import re

with open('chapters/chapter05.tex', 'r') as f:
    content = f.read()

# We need to find all algorithm figures.
# The user specified 8 algorithms: 
# alg:homography_svd, alg:scale_normalization, alg:lstm_touch_inference, alg:coordinate_mapping, 
# alg:canvas_drag_drop, alg:queue_synchronization, alg:resample_12fps, alg:camera_calibration

algs = [
    'alg:homography_svd', 'alg:scale_normalization', 'alg:lstm_touch_inference', 
    'alg:coordinate_mapping', 'alg:canvas_drag_drop', 'alg:queue_synchronization', 
    'alg:resample_12fps', 'alg:camera_calibration'
]

def process_figure(match):
    fig_text = match.group(0)
    if not any(a in fig_text for a in algs):
        return fig_text
    
    # Apply replacements
    # 1. wrap in minipage
    # First, let's see where to put minipage. \centering\n\begin{tabular}... \end{tabular}
    # It might be easier to replace \begin{tabular} with \begin{minipage}{0.92\textwidth}\n\small\n\setlength{\extrarowheight}{2pt}\n\begin{tabular}
    
    # Change tabular column from |p{0.95\textwidth}| to |p{0.88\textwidth}|
    fig_text = fig_text.replace(r'\begin{tabular}{|p{0.95\textwidth}|}', 
r'''\begin{minipage}{0.92\textwidth}
\small
\setlength{\extrarowheight}{2pt}
\begin{tabular}{|p{0.88\textwidth}|}''')
    
    fig_text = fig_text.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + r'\end{minipage}')

    # Input/Output
    fig_text = re.sub(r'\\textbf{Input:} (.*?)( \\\\)', r'{\\small\\sffamily \\textbf{Input:} \1}\2', fig_text)
    fig_text = re.sub(r'\\textbf{Output:} (.*?)( \\\\)', r'{\\small\\sffamily \\textbf{Output:} \1}\2', fig_text)
    
    # Numbered steps
    # We want to add \small to each numbered step.
    # Lines starting with digits e.g. "1: " or "10: "
    # We can match `^(\d+:.*?)( \\\\)` with multiline
    
    def replace_step(m):
        step_content = m.group(1)
        # replace \quad and \qquad
        step_content = step_content.replace(r'\qquad', r'\hspace{2em}')
        step_content = step_content.replace(r'\quad', r'\hspace{1em}')
        return r'{\small ' + step_content + r'}' + m.group(2)
        
    fig_text = re.sub(r'^(\d+:\s.*?)( \\\\)$', replace_step, fig_text, flags=re.MULTILINE)
    
    return fig_text

new_content = re.sub(r'\\begin{figure}\[htbp\].*?\\end{figure}', process_figure, content, flags=re.DOTALL)

with open('chapters/chapter05.tex', 'w') as f:
    f.write(new_content)

print("Done")
