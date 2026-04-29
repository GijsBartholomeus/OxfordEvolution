from __future__ import annotations

from pathlib import Path


MAIN = Path("/Users/gijsbartholomeus/Documents/STUDIE/Papers/WhySystemsBiologyWorks/main.tex")


MODEL_TABLE = r"""\begin{table}[!t]
    \centering
    \small
    \caption{\textbf{Seven oscillatory ODE models used for the genotype--phenotype-map analysis.}
    For each model we list the biological process, the key output molecule used to define the oscillation phenotype, the number of dynamic variables $m$, and the number of varied parameters $N$ after excluding unsettable, zero-valued, or switch-like quantities.}
    \label{tab:models}
    \begin{tabular}{llrr}
        \hline
        Model & Output & $m$ & $N$ \\
        \hline
        Chen 2004 \cite{Chen2004} & CLB2 & 50 & 136 \\
        Kholodenko 2000 \cite{Kholodenko2000} & MKK\_PP & 8 & 21 \\
        Leloup 1999 \cite{Leloup1999} & Cn & 10 & 37 \\
        Locke 2005 \cite{Locke2005} & cXn & 13 & 61 \\
        Ueda 2001 \cite{Ueda2001} & CCc & 10 & 46 \\
        Vilar 2002 \cite{Vilar2002} & C & 9 & 15 \\
        Tyson 1991 \cite{Tyson1991} & M & 6 & 8 \\
        \hline
    \end{tabular}
\end{table}"""


FIG1 = r"""\begin{figure}[!t]
    \centering
    \includegraphics[width=\linewidth]{Figures/NeutralGeometryCartoon.png}
    \caption{\textbf{Two candidate geometries for the wild-type neutral set.}
    \textbf{(A)} A localized, compact region. A random initial parameter vector (red cross) is typically far from the region, and gradient-based fitting from the cross has no guarantee of reaching it.
    \textbf{(B)} An extended structure of the same total volume. Random initial points are generically close to some part of the structure, and motion along sloppy directions (arrows) reaches it quickly. The two panels have the same neutral-set size and very different accessibility.}
    \label{fig:cartoon}
\end{figure}"""


FIG2 = r"""\begin{figure*}[!t]
    \centering
    \includegraphics[width=\textwidth]{Figures/FreqComp.png}
    \caption{\textbf{Genotype--phenotype maps for gene regulatory networks are biased towards simple phenotypes.}
    \textbf{(A)} Scatter plot of phenotype frequency $P(x)$ versus phenotype complexity $K(x)$ for the budding yeast cell-cycle model of Chen et al.~\cite{Chen2004}. The purple hull around the majority of points is a visual aid for interpreting \textbf{(B)}, which shows the corresponding $P(x)$--$K(x)$ hulls for six additional oscillatory gene-regulatory and signalling systems: \textbf{(i)} the MAPK cascade model of Kholodenko~\cite{Kholodenko2000}, \textbf{(ii)} the Drosophila circadian clock model of Leloup and Goldbeter~\cite{Leloup1999}, \textbf{(iii)} the Arabidopsis circadian clock model of Locke et al.~\cite{Locke2005}, \textbf{(iv)} the mammalian circadian clock model of Ueda et al.~\cite{Ueda2001}, \textbf{(v)} the synthetic genetic oscillator model of Vilar et al.~\cite{Vilar2002}, and \textbf{(vi)} the cdc2--cyclin cell-cycle model of Tyson~\cite{Tyson1991}. For all systems, the oscillation phenotype genotype--phenotype map shows a strong bias towards low-complexity phenotypes. The wild-type oscillatory phenotype is shown as a red dot where it falls within the observed frequency range; in panel \textbf{(A)} and subpanels \textbf{B(i)} and \textbf{B(iv)} it is omitted for visual clarity because it was not observed in the finite sample and would otherwise be plotted at the pseudo-frequency $0.5/N$.}
    \label{fig:designability}
\end{figure*}"""


def replace_block(text: str, label: str, replacement: str) -> str:
    marker = rf"\label{{{label}}}"
    label_pos = text.index(marker)
    begin = text.rfind(r"\begin{figure}", 0, label_pos)
    begin_star = text.rfind(r"\begin{figure*}", 0, label_pos)
    if begin_star > begin:
        begin = begin_star
    end = text.index(r"\end{figure", label_pos)
    end = text.index("}", end) + 1
    return text[:begin] + replacement + text[end:]


def replace_table_block(text: str, label: str, replacement: str) -> str:
    marker = rf"\label{{{label}}}"
    label_pos = text.index(marker)
    begin = text.rfind(r"\begin{table}", 0, label_pos)
    begin_star = text.rfind(r"\begin{table*}", 0, label_pos)
    if begin_star > begin:
        begin = begin_star
    end = text.index(r"\end{table", label_pos)
    end = text.index("}", end) + 1
    return text[:begin] + replacement + text[end:]


def main() -> None:
    text = MAIN.read_text()

    text = replace_block(text, "fig:cartoon", FIG1)
    text = replace_block(text, "fig:designability", FIG2)

    old_table_note = r"\note{Table 1: list of 12 ODE models with key output molecule, number of parameters $N$, number of variables $m$.}"
    if old_table_note in text:
        text = text.replace(old_table_note, MODEL_TABLE)
    elif r"\label{tab:models}" in text:
        text = replace_table_block(text, "tab:models", MODEL_TABLE)

    text = text.replace("Each of the twelve models (Table~\\ref{tab:models})", "Each of the seven models (Table~\\ref{tab:models})")
    text = text.replace("For each of the twelve models, we sampled", "For each of the seven models, we sampled")
    text = text.replace("Across the twelve models", "Across the seven models")
    text = text.replace("for all twelve models", "for all seven models")
    text = text.replace("all twelve models", "all seven models")
    text = text.replace("the twelve models", "the seven models")
    text = text.replace("The twelve ODE models used in this study", "The seven ODE models used in this study")
    text = text.replace("twelve benchmark ordinary differential equation models", "seven oscillatory ordinary differential equation models")
    text = text.replace("twelve benchmark models", "seven oscillatory models")
    text = text.replace("twelve benchmark ODE models", "seven oscillatory ODE models")
    text = text.replace("the other eleven models", "the other six models")
    text = text.replace("remaining eleven models", "remaining six models")
    text = text.replace("all 12 models", "all seven models")
    text = text.replace("the 12 models", "the seven models")
    text = text.replace("For each of the 12 models", "For each of the seven models")
    text = text.replace("at six biochemical parameters it is small enough", "at six dynamic variables and eight varied biochemical parameters it is small enough")

    MAIN.write_text(text)


if __name__ == "__main__":
    main()
