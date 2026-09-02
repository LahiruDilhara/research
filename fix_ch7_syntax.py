import os

with open('chapters/chapter07.tex', 'r') as f:
    text = f.read()

# Fix unclosed itemize in 7.3.4
text = text.replace(r"\item \textbf{Technical Mitigation:} Integrated immediate low-latency visual and auditory feedback dispatches into Application 2 Component B. Upon touch confirmation ($p_k > 0.90$), the system plays a subtle, short audio click ($< 5\text{ ms}$ latency) and briefly flashes a green visual highlight on the corresponding key button graphic in the live UI overlay." + "\n\n" + r"\section{Self-Reflection}",
r"\item \textbf{Technical Mitigation:} Integrated immediate low-latency visual and auditory feedback dispatches into Application 2 Component B. Upon touch confirmation ($p_k > 0.90$), the system plays a subtle, short audio click ($< 5\text{ ms}$ latency) and briefly flashes a green visual highlight on the corresponding key button graphic in the live UI overlay." + "\n" + r"\end{itemize}" + "\n\n" + r"\section{Self-Reflection}")

# Fix & escape in section 7.4.2
text = text.replace(r"\item \textbf{Empirical Research & LaTeX Documentation:}", r"\item \textbf{Empirical Research \& LaTeX Documentation:}")

# Fix markdown header syntax \##
text = text.replace(r"\## Low-Cost Educational Classrooms in Developing Regions", r"\subsection{Low-Cost Educational Classrooms in Developing Regions}")

with open('chapters/chapter07.tex', 'w') as f:
    f.write(text)

print("Fixed syntax issues in chapter07.tex!")
