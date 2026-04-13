import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

logger = logging.getLogger(__name__)

DOW_LABELS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


class Plotter:
    """
    Genera todas las visualizaciones del pipeline.

    Parámetros
    ──────────
    cfg     : dict completo del config.yaml
    out_dir : directorio de salida para las figuras
    """

    def __init__(self, cfg: dict, out_dir: str):
        self.cfg     = cfg
        self.out_dir = out_dir
        self.colors  = cfg["visualization"]["colors"]
        self.dpi     = cfg["visualization"]["dpi"]
        os.makedirs(out_dir, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        bg = self.colors["bg"]
        plt.rcParams.update({
            "figure.facecolor" : bg,
            "axes.facecolor"   : bg,
            "axes.spines.top"  : False,
            "axes.spines.right": False,
            "font.family"      : "DejaVu Sans",
            "axes.titlesize"   : 11,
            "axes.labelsize"   : 10,
        })

    def _save(self, fig, filename: str):
        path = os.path.join(self.out_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figura guardada: %s", path)
        return path

    # ── 1. Visión general ────────────────────────────────────────────────

    def plot_eda_overview(self, df: pd.DataFrame):
            
        PALETTE = ["#D61D6B", "#6B21A8", "#F472B6", "#2D1B4E", "#DB2777",
           "#9333EA", "#EC4899", "#7C3AED", "#BE185D", "#5B21B6",
           "#A21CAF", "#86198F", "#701A75", "#4A044E", "#3B0764"]
        BLUE   = self.colors["blue"]
        RED    = self.colors["red"]
        GREEN  = self.colors["green"]
        ORANGE = self.colors["orange"]

        daily_total = df.groupby("fecha_dia").size().reset_index(name="n_tx")
        ma7         = daily_total["n_tx"].rolling(7, center=True).mean()
        lunes       = pd.date_range(
            daily_total["fecha_dia"].min(),
            daily_total["fecha_dia"].max(),
            freq="W-MON"
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("EDA — Visión General del Dataset Transaccional",
                    fontsize=15, fontweight="bold")

        # ── Panel superior izquierdo: volumen diario con bandas semanales ─
        ax = axes[0, 0]

        # Bandas alternas por semana
        for k, lun in enumerate(lunes[:-1]):
            fin   = lunes[k + 1]
            color = "#DBEAFE" if k % 2 == 0 else "#EFF6FF"
            ax.axvspan(lun, fin, alpha=0.35, color=color, zorder=0)
            centro = lun + (fin - lun) / 2
            ax.text(centro, daily_total["n_tx"].max() * 1.01,
                    f"S{k+1}", ha="center", va="bottom",
                    fontsize=6, color="#64748B", fontweight="bold")

        # Líneas punteadas cada lunes
        for lun in lunes:
            ax.axvline(lun, color="#94A3B8", lw=0.8, ls="--", zorder=1, alpha=0.8)

        # Serie diaria y MA-7
        ax.fill_between(daily_total["fecha_dia"], daily_total["n_tx"],
                        alpha=0.20, color=BLUE, zorder=2)
        ax.plot(daily_total["fecha_dia"], daily_total["n_tx"],
                color=BLUE, lw=1.5, label="Tx diarias", zorder=3)
        ax.plot(daily_total["fecha_dia"], ma7,
                color=RED, lw=1.5, ls="--", label="MA-7", zorder=4)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)
        ax.set_title("Volumen diario total de transacciones")
        ax.set_ylabel("# Transacciones / día")
        ax.set_ylim(0, daily_total["n_tx"].max() * 1.08)
        ax.legend(fontsize=8)

        # ── Panel superior derecho: top 15 operaciones ────────────────────
        ax = axes[0, 1]
        oper_counts = df["oper"].value_counts().head(15)
        bars = ax.bar(
            oper_counts.index.astype(str),
            oper_counts.values,
            color=PALETTE[:len(oper_counts)]
        )
        ax.set_title("Top 15 operaciones por volumen total")
        ax.set_xlabel("Código de operación")
        ax.set_ylabel("# Transacciones")
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 500,
                f"{b.get_height():,.0f}",
                ha="center", va="bottom", fontsize=7.5
            )

        # ── Panel inferior izquierdo: top 20 terminales ───────────────────
        ax = axes[1, 0]
        term_counts = df["idTerminal"].value_counts().head(20)
        ax.barh(
            term_counts.index.astype(str)[::-1],
            term_counts.values[::-1],
            color=GREEN, alpha=0.85
        )
        ax.set_title("Top 20 terminales por volumen total")
        ax.set_xlabel("# Transacciones")
        ax.set_ylabel("Terminal ID")
        for i, v in enumerate(term_counts.values[::-1]):
            ax.text(v + 200, i, f"{v:,}", va="center", fontsize=7.5)

        # ── Panel inferior derecho: volumen por día de semana ─────────────
        ax = axes[1, 1]
        dow_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        dow_counts  = df["dow"].value_counts().sort_index()
        dow_colors = ["#D61D6B" if i < 5 else "#2D1B4E" for i in range(7)]
        ax.bar(dow_labels, dow_counts.values, color=dow_colors, alpha=0.85)
        ax.set_title("Volumen total por día de la semana")
        ax.set_ylabel("# Transacciones")
        ax.set_xlabel("Día")

        fig.tight_layout()
        return self._save(fig, "fig1_eda_overview.png")

    # ── 2. Patrones temporales ───────────────────────────────────────────

    def plot_temporal_patterns(self, df: pd.DataFrame):
        """Fig 2 — Heatmap hora×DOW y volumen horario promedio."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("EDA — Patrones Temporales", fontsize=15, fontweight="bold")

        # Heatmap
        ax = axes[0]
        pivot = (
            df.groupby(["dow", "hora"]).size()
            .reset_index(name="n")
            .pivot(index="dow", columns="hora", values="n")
            .fillna(0)
        )
        pivot.index = DOW_LABELS
        sns.heatmap(pivot, ax=ax, cmap="RdPu", linewidths=0.3,
                    cbar_kws={"label": "# Transacciones", "shrink": 0.8})
        ax.set_title("Heatmap: Día de semana × Hora del día")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Día de la semana")

        # Volumen horario
        ax = axes[1]
        hourly_avg = df.groupby("hora").size() / df["fecha_dia"].nunique()
        ax.bar(hourly_avg.index, hourly_avg.values,
               color=self.colors["blue"], alpha=0.85)
        ax.set_title("Promedio de transacciones por hora del día")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Promedio diario de transacciones")
        ax.set_xticks(range(0, 24))

        fig.tight_layout()
        return self._save(fig, "fig2_temporal_patterns.png")

    # ── 3. Series por combinación ────────────────────────────────────────

    def plot_series_grid(
        self,
        daily     : pd.DataFrame,
        terminals : list,
        opers     : list,
    ):
        """Fig 3 — Grilla 5×5 de series históricas."""
        PALETTE = ["#D61D6B", "#6B21A8", "#F472B6", "#2D1B4E", "#DB2777",
           "#9333EA", "#EC4899", "#7C3AED", "#BE185D", "#5B21B6"]
        fig, axes = plt.subplots(5, 5, figsize=(22, 18), sharex=True)
        fig.suptitle("Series de tiempo diarias — Top 5 terminales × Top 5 operaciones",
                     fontsize=14, fontweight="bold")

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax = axes[i, j]
                sub = daily[
                    (daily["idTerminal"] == term) & (daily["oper"] == op)
                ]
                ax.plot(sub["fecha_dia"], sub["n_tx"],
                        lw=1.2, color=PALETTE[j], alpha=0.9)
                ax.fill_between(sub["fecha_dia"], sub["n_tx"],
                                alpha=0.15, color=PALETTE[j])
                ax.set_ylabel("# Tx / día", fontsize=7)
                if i == 0:
                    ax.set_title(f"Oper {op}", fontsize=10, color=PALETTE[j])
                if j == 0:
                    ax.set_ylabel(f"T-{term}", fontsize=8,
                                  rotation=0, labelpad=42, va="center")
                ax.tick_params(labelsize=6)
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        fig.tight_layout()
        return self._save(fig, "fig3_series_grid.png")

    # ── 4. Comparación de modelos — boxplots ─────────────────────────────

    def plot_model_comparison_boxplots(self, df_metrics: pd.DataFrame):
        """Fig 4 — Boxplots MAE/RMSE/MAPE por modelo."""
        MODEL_COLORS = {
        "Prophet"    : "#D61D6B",   # magenta
        "SARIMA"     : "#6B21A8",   # morado
        "HoltWinters": "#F472B6",   # rosa
        }
        models = ["Prophet", "SARIMA", "HoltWinters"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Comparación de modelos — Distribución de métricas (25 series)",
                     fontsize=13, fontweight="bold")

        for ax, metric, ylabel in zip(
            axes,
            ["MAE", "RMSE", "MAPE_%"],
            ["MAE (tx/día)", "RMSE (tx/día)", "MAPE (%)"],
        ):
            data_bp = [df_metrics[df_metrics["modelo"] == m][metric].values
                       for m in models]
            bp = ax.boxplot(
                data_bp, patch_artist=True,
                medianprops={"color": "black", "lw": 2},
            )
            for patch, m in zip(bp["boxes"], models):
                patch.set_facecolor(MODEL_COLORS[m])
                patch.set_alpha(0.75)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["Prophet", "SARIMA", "Holt-\nWinters"])
            ax.set_title(metric)
            ax.set_ylabel(ylabel)
            for k, d in enumerate(data_bp):
                med = np.median(d)
                ax.text(k + 1, med, f" {med:.1f}", va="bottom",
                        fontsize=8.5, fontweight="bold")

        fig.tight_layout()
        return self._save(fig, "fig4_model_comparison_boxplots.png")

    # ── 5. Heatmap MAE por combinación ───────────────────────────────────

    def plot_mae_heatmaps(self, df_metrics: pd.DataFrame):
        """Fig 5 — Heatmap MAE por combinación para cada modelo."""
        MODEL_COLORS = {
        "Prophet"    : "#D61D6B",   # magenta
        "SARIMA"     : "#6B21A8",   # morado
        "HoltWinters": "#F472B6",   # rosa
        }
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("MAE por combinación (terminal × operación) — cada modelo",
                     fontsize=13, fontweight="bold")

        active_models = df_metrics["modelo"].unique().tolist()
        fig, axes = plt.subplots(1, len(active_models), figsize=(6*len(active_models), 6))
        if len(active_models) == 1: axes = [axes]
        for ax, modelo in zip(axes, active_models):
            pivot = df_metrics[df_metrics["modelo"] == modelo].pivot_table(
                index="idTerminal", columns="oper", values="MAE"
            )
            sns.heatmap(pivot, ax=ax, cmap="RdPu",
                        annot=True, fmt=".1f", linewidths=0.4,
                        cbar_kws={"label": "MAE", "shrink": 0.8})
            ax.set_title(modelo, fontweight="bold",
                         color=MODEL_COLORS[modelo])
            ax.set_xlabel("Operación")
            ax.set_ylabel("Terminal ID" if ax == axes[0] else "")

        fig.tight_layout()
        return self._save(fig, "fig5_mae_heatmaps.png")

    # ── 6. Victorias por modelo ───────────────────────────────────────────

    def plot_winner_summary(
        self,
        df_winners : pd.DataFrame,
        df_metrics : pd.DataFrame,
        top_opers  : list,
    ):
        """Fig 6 — Donut de victorias + MAE promedio por operación."""
        MODEL_COLORS = {
        "Prophet"    : "#D61D6B",   # magenta
        "SARIMA"     : "#6B21A8",   # morado
        "HoltWinters": "#F472B6",   # rosa
        }
        wins = df_winners["mejor_modelo"].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Selección de modelo ganador por serie",
                     fontsize=13, fontweight="bold")

        # Donut
        ax = axes[0]
        labels = wins.index.tolist()
        cols   = [MODEL_COLORS[l] for l in labels]
        _, _, autotexts = ax.pie(
            wins.values, labels=labels, colors=cols, autopct="%1.0f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops={"width": 0.55, "edgecolor": "white", "lw": 2},
        )
        for at in autotexts:
            at.set_fontsize(12); at.set_fontweight("bold")
        ax.set_title("Victorias (menor MAE en holdout)")

        # Barras por operación
        ax = axes[1]
        mae_op = df_metrics.groupby(["oper", "modelo"])["MAE"].mean().reset_index()
        active_m = df_metrics["modelo"].unique().tolist()
        x = np.arange(len(top_opers))
        w = 0.8 / len(active_m)
        for k, modelo in enumerate(active_m):
            vals = [
                mae_op[(mae_op["oper"] == op) & (mae_op["modelo"] == modelo)]["MAE"].values[0]
                for op in top_opers
            ]
            ax.bar(x + k * w, vals, w, label=modelo,
                   color=MODEL_COLORS[modelo], alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels([f"Oper {o}" for o in top_opers])
        ax.set_ylabel("MAE promedio (5 terminales)")
        ax.set_title("MAE promedio por operación y modelo")
        ax.legend(fontsize=9)

        fig.tight_layout()
        return self._save(fig, "fig6_winner_summary.png")

    # ── 7. Holdout: 3 modelos vs real ─────────────────────────────────────

    def plot_holdout_all_models(
        self,
        df_preds  : pd.DataFrame,
        df_winners: pd.DataFrame,
        terminals : list,
        opers     : list,
    ):
        """Fig 7 — Grilla 5×5: Real vs predicción de los 3 modelos en holdout."""
        MODEL_COLORS = {
        "Prophet"    : "#D61D6B",   # magenta
        "SARIMA"     : "#6B21A8",   # morado
        "HoltWinters": "#F472B6",   # rosa
         }
        GRAY = self.colors["gray"]

        fig, axes = plt.subplots(5, 5, figsize=(22, 17))
        fig.suptitle(
            "Holdout 15 días — Predicción vs Real (3 modelos)\n"
            "Azul=Prophet  Verde=SARIMA  Naranja=Holt-Winters  Negro=Real",
            fontsize=12, fontweight="bold",
        )

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax = axes[i, j]
                real = (
                    df_preds[
                        (df_preds["idTerminal"] == term) &
                        (df_preds["oper"] == op) &
                        (df_preds["modelo"] == "Prophet")
                    ][["ds", "y_real"]].sort_values("ds")
                )
                ax.plot(real["ds"], real["y_real"],
                        "k-o", ms=3, lw=1.5, label="Real", zorder=5)

                for modelo, col in MODEL_COLORS.items():
                    sub = df_preds[
                        (df_preds["idTerminal"] == term) &
                        (df_preds["oper"] == op) &
                        (df_preds["modelo"] == modelo)
                    ].sort_values("ds")
                    ax.plot(sub["ds"], sub["y_pred"],
                            "--", color=col, lw=1.3, ms=2.5,
                            marker="s", alpha=0.85, label=modelo)

                w_row = df_winners[
                    (df_winners["idTerminal"] == term) &
                    (df_winners["oper"] == op)
                ]
                best = w_row["mejor_modelo"].values[0] if not w_row.empty else "?"
                ax.set_title(f"T{term}/Op{op} ★{best[:4]}", fontsize=7.5)
                ax.tick_params(labelsize=5.5)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                if i == 0 and j == 0:
                    ax.legend(fontsize=5.5, ncol=2)

        fig.tight_layout()
        return self._save(fig, "fig7_holdout_all_models.png")

    # ── 8. HW con IC empírico 95% ─────────────────────────────────────────

    def plot_hw_ic95(
        self,
        daily    : pd.DataFrame,
        hw_results: dict,
        terminals: list,
        opers    : list,
    ):
        """
        Fig 8 — Holt-Winters predicción vs real con IC 95% empírico.

        hw_results : dict keyed by (term, op) con valores:
            {pred, lower, upper, test_y, test_idx, mae, cobertura, n_folds}
        """
        BLUE   = self.colors["blue"]
        ORANGE = self.colors["orange"]

        fig, axes = plt.subplots(5, 5, figsize=(22, 18))
        fig.supylabel("# Transacciones / día", fontsize=11, x=0.0)
        fig.suptitle(
            "Holt-Winters — Predicción vs Real · Holdout 15 días\n"
            "Banda: IC 95% empírico (validación cruzada por bloques deslizantes)",
            fontsize=13, fontweight="bold", y=1.005,
        )

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax    = axes[i, j]
                res   = hw_results[(term, op)]
                pred  = res["pred"]
                lower = res["lower"]
                upper = res["upper"]
                test_y   = res["test_y"]
                test_idx = res["test_idx"]

                ax.fill_between(test_idx, lower, upper,
                                color=ORANGE, alpha=0.30, label="IC 95% emp.")
                ax.plot(test_idx, lower, color=ORANGE, lw=0.8, ls=":", alpha=0.7)
                ax.plot(test_idx, upper, color=ORANGE, lw=0.8, ls=":", alpha=0.7)
                ax.plot(test_idx, pred,  color=ORANGE, lw=1.8,
                        marker="s", ms=3.5, label="HW pred")
                ax.plot(test_idx, test_y, color=BLUE, lw=1.6,
                        marker="o", ms=3.5, label="Real")

                ax.set_title(
                    f"T{term}/Op{op}\n"
                    f"MAE={res['mae']:.1f}  Cob={res['cobertura']:.0f}%  "
                    f"(k={res['n_folds']})",
                    fontsize=7,
                )
                ax.tick_params(labelsize=5.5)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6.5, loc="upper left",
                              framealpha=0.7, handlelength=1.5)

        fig.tight_layout()
        return self._save(fig, "fig8_hw_ic95_empirico.png")

    # ── 9. Pronóstico futuro con modelo ganador ───────────────────────────

    def plot_forecast_winners(
        self,
        daily    : pd.DataFrame,
        df_future: pd.DataFrame,
        terminals: list,
        opers    : list,
    ):
        """Fig 9 — Pronóstico 15 días con modelo ganador por serie."""
        MODEL_COLORS = {
        "Prophet"    : "#D61D6B",   # magenta
        "SARIMA"     : "#6B21A8",   # morado
        "HoltWinters": "#F472B6",   # rosa
        }
        GRAY = self.colors["gray"]

        fig, axes = plt.subplots(5, 5, figsize=(22, 17))
        fig.supylabel("# Transacciones / día", fontsize=11, x=0.0)
        fig.suptitle(
            "Pronóstico 15 días (2017-06-01 → 2017-06-15)\n"
            "Modelo ganador por serie (menor MAE en holdout) | IC 95% empírico",
            fontsize=12, fontweight="bold",
        )

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax   = axes[i, j]
                hist = daily[
                    (daily["idTerminal"] == term) & (daily["oper"] == op)
                ].sort_values("fecha_dia").tail(30)
                fwd = df_future[
                    (df_future["idTerminal"] == term) & (df_future["oper"] == op)
                ].sort_values("ds")

                if fwd.empty:
                    ax.axis("off"); continue

                best  = fwd["modelo_ganador"].iloc[0]
                color = MODEL_COLORS.get(best, GRAY)

                ax.plot(hist["fecha_dia"], hist["n_tx"],
                        color=GRAY, lw=1.2, alpha=0.7, label="Histórico")
                ax.axvline(pd.Timestamp("2017-05-31"),
                           color=GRAY, ls=":", lw=0.8)
                ax.plot(fwd["ds"], fwd["yhat"],
                        color=color, lw=1.8, marker=".", ms=4, label=best)
                ax.fill_between(fwd["ds"], fwd["yhat_lower"], fwd["yhat_upper"],
                                color=color, alpha=0.22, label="IC 95%")

                ax.set_title(f"T{term}/Op{op}\n[{best[:4]}]",
                             fontsize=7.5, color=color)
                ax.tick_params(labelsize=5)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)

        fig.tight_layout()
        return self._save(fig, "fig9_forecast_winners.png")

    # ── 10. Heatmap demanda predicha ──────────────────────────────────────

    def plot_demand_heatmap(self, df_future: pd.DataFrame):
        """Fig 10 — Heatmap de demanda total predicha por terminal y día."""
        pivot = (
            df_future.groupby(["idTerminal", "ds"])["yhat"]
            .sum().reset_index()
            .pivot(index="idTerminal", columns="ds", values="yhat")
        )
        pivot.columns = [
            pd.Timestamp(c).strftime("%d-%b") for c in pivot.columns
        ]
        fig, ax = plt.subplots(figsize=(16, 4))
        fig.suptitle(
            "Demanda total predicha por terminal (suma 5 operaciones)\n"
            "2017-06-01 → 2017-06-15  |  # Transacciones / día",
            fontsize=12, fontweight="bold",
        )
        sns.heatmap(pivot, ax=ax, cmap="RdPu",
                    annot=True, fmt=".0f", linewidths=0.4,
                    cbar_kws={"label": "# Tx predichas", "shrink": 0.6})
        ax.set_xlabel("Día")
        ax.set_ylabel("Terminal ID")
        fig.tight_layout()
        return self._save(fig, "fig10_demand_heatmap.png")
    
    # ── 11. STL serie total ───────────────────────────────────────────────

    def plot_stl_total(self, df: pd.DataFrame):
        """Fig 11 — Descomposicion STL de la serie total."""
        from statsmodels.tsa.seasonal import STL

        MAGENTA = self.colors["blue"]    # magenta Avista
        PURPLE  = self.colors["red"]     # morado Avista
        PINK    = self.colors["green"]   # rosa Avista

        serie = df.groupby("fecha_dia").size().reset_index(name="n_tx") \
                .set_index("fecha_dia")["n_tx"].asfreq("D")

        stl    = STL(serie, period=7, robust=True)
        result = stl.fit()

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.suptitle("Descomposicion STL — Serie total de transacciones diarias\n"
                    "period=7 | robust=True",
                    fontsize=14, fontweight="bold", color=PURPLE)

        componentes = [
            (result.observed, "Serie observada", MAGENTA),
            (result.trend,    "Tendencia",        PURPLE),
            (result.seasonal, "Estacionalidad",   PINK),
            (result.resid,    "Residuo",          MAGENTA),
        ]
        for ax, (comp, titulo, color) in zip(axes, componentes):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if titulo == "Residuo":
                ax.bar(comp.index, comp.values, color=color, alpha=0.6, width=1)
                ax.axhline(0, color=self.colors["gray"], lw=0.8, ls="--")
            else:
                ax.fill_between(comp.index, comp.values, alpha=0.15, color=color)
                ax.plot(comp.index, comp.values, color=color, lw=1.8)
            ax.set_ylabel(titulo, fontsize=10, color=PURPLE)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))

        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        return self._save(fig, "fig11_stl_total.png")


    # ── 12. STL 5x5 ──────────────────────────────────────────────────────

    def plot_stl_5x5(
        self,
        daily    : pd.DataFrame,
        terminals: list,
        opers    : list,
    ):
        """Fig 12 — STL grilla 5x5 para las 25 combinaciones."""
        from statsmodels.tsa.seasonal import STL

        MAGENTA = self.colors["blue"]
        PURPLE  = self.colors["red"]
        PINK    = self.colors["green"]
        PALETTE = [self.colors["blue"], self.colors["red"],
                self.colors["green"], self.colors["orange"],
                self.colors["gray"]]

        fig, axes = plt.subplots(5, 5, figsize=(24, 20))
        fig.suptitle("Descomposicion STL por combinacion (terminal x operacion)\n"
                    "Magenta=Observada  Morado=Tendencia  Rosa=Estac+Tend",
                    fontsize=13, fontweight="bold", color=PURPLE)

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax = axes[i, j]
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                sub = (
                    daily[(daily["idTerminal"]==term) & (daily["oper"]==op)]
                    .sort_values("fecha_dia")
                    .set_index("fecha_dia")["n_tx"]
                    .asfreq("D")
                )
                try:
                    res = STL(sub, period=7, robust=True).fit()
                    ax.plot(sub.index, sub.values,
                            color=MAGENTA, lw=0.8, alpha=0.4, label="Observada")
                    ax.plot(res.trend.index, res.trend.values,
                            color=PURPLE, lw=1.8, label="Tendencia")
                    ax.plot((res.seasonal + res.trend).index,
                            (res.seasonal + res.trend).values,
                            color=PINK, lw=1.0, alpha=0.7, ls="--", label="Estac+Tend")
                except Exception:
                    ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="gray")

                ax.set_title(f"T{term}/Op{op}", fontsize=8, fontweight="bold",
                            color=PALETTE[j % len(PALETTE)])
                ax.tick_params(labelsize=5)
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                if i == 0 and j == 0:
                    ax.legend(fontsize=5.5, loc="upper left")

        fig.tight_layout()
        return self._save(fig, "fig12_stl_5x5.png")


    # ── 13. ACF diferenciada serie total ─────────────────────────────────

    def plot_acf_total(self, df: pd.DataFrame):
        """Fig 13 — ACF diferenciada (d=1) de la serie total."""
        from statsmodels.graphics.tsaplots import plot_acf

        MAGENTA = self.colors["blue"]
        PURPLE  = self.colors["red"]
        RED     = self.colors["orange"]

        serie      = df.groupby("fecha_dia").size().reset_index(name="n_tx") \
                    .set_index("fecha_dia")["n_tx"].asfreq("D")
        serie_diff = serie.diff().dropna()

        fig, ax = plt.subplots(figsize=(16, 5))
        fig.suptitle("ACF serie diferenciada (d=1) — Serie total\n"
                    "Lineas rojas = lags 7, 14, 21 | Banda = IC 95%",
                    fontsize=13, fontweight="bold", color=PURPLE)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plot_acf(serie_diff, lags=35, ax=ax,
                color=MAGENTA,
                vlines_kwargs={"colors": MAGENTA},
                alpha=0.05, title="")

        ylim = ax.get_ylim()
        for lag in [7, 14, 21]:
            ax.axvline(lag, color=RED, lw=1.2, ls="--", alpha=0.85, zorder=5)
            ax.text(lag, ylim[1] * 0.90, f"lag {lag}",
                    ha="center", fontsize=8.5, color=RED, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                            fc="white", ec=RED, alpha=0.85))

        ax.axhline(0, color=self.colors["gray"], lw=0.6)
        ax.set_xlabel("Lag (dias)", fontsize=10)
        ax.set_ylabel("Autocorrelacion", fontsize=10)

        fig.tight_layout()
        return self._save(fig, "fig13_acf_total.png")


    # ── 14. ACF diferenciada 5x5 ─────────────────────────────────────────

    def plot_acf_5x5(
        self,
        daily    : pd.DataFrame,
        terminals: list,
        opers    : list,
    ):
        """Fig 14 — ACF diferenciada (d=1) grilla 5x5."""
        from statsmodels.graphics.tsaplots import plot_acf

        MAGENTA = self.colors["blue"]
        PURPLE  = self.colors["red"]
        RED     = self.colors["orange"]
        PALETTE = [self.colors["blue"], self.colors["red"],
                self.colors["green"], self.colors["orange"],
                self.colors["gray"]]

        fig, axes = plt.subplots(5, 5, figsize=(24, 20))
        fig.suptitle("ACF diferenciada (d=1) — 5x5 combinaciones\n"
                    "Lineas rojas = lags 7, 14, 21 | Banda = IC 95%",
                    fontsize=13, fontweight="bold", color=PURPLE)

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax = axes[i, j]
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                color = PALETTE[j % len(PALETTE)]

                sub = (
                    daily[(daily["idTerminal"]==term) & (daily["oper"]==op)]
                    .sort_values("fecha_dia")
                    .set_index("fecha_dia")["n_tx"]
                    .asfreq("D").fillna(0)
                )
                sub_diff = sub.diff().dropna()

                try:
                    plot_acf(sub_diff, lags=28, ax=ax,
                            color=color,
                            vlines_kwargs={"colors": color},
                            alpha=0.05, title="")

                    ylim = ax.get_ylim()
                    for lag in [7, 14, 21]:
                        ax.axvline(lag, color=RED, lw=1.0, ls="--", alpha=0.85)
                        ax.text(lag, ylim[1] * 0.88, f"{lag}",
                                ha="center", fontsize=6.5,
                                color=RED, fontweight="bold")
                    ax.axhline(0, color=self.colors["gray"], lw=0.6)

                except Exception:
                    ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="gray")

                ax.set_title(f"T{term}/Op{op}", fontsize=8,
                            fontweight="bold", color=color)
                ax.set_xlabel("Lag", fontsize=6)
                ax.set_ylabel("ACF", fontsize=6)
                ax.tick_params(labelsize=5.5)

        fig.tight_layout()
        return self._save(fig, "fig14_acf_5x5.png")


    # ── 15. Promedio por dia de la semana 5x5 ────────────────────────────

    def plot_dow_5x5(
        self,
        daily    : pd.DataFrame,
        terminals: list,
        opers    : list,
    ):
        """Fig 15 — Promedio de transacciones por dia de la semana, grilla 5x5."""
        MAGENTA = self.colors["blue"]
        PURPLE  = self.colors["red"]
        PALETTE = [self.colors["blue"], self.colors["red"],
                self.colors["green"], self.colors["orange"],
                self.colors["gray"]]

        dow_labels  = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
        dow_colors  = [MAGENTA] * 5 + [PURPLE] * 2

        daily = daily.copy()
        daily["dow"] = pd.to_datetime(daily["fecha_dia"]).dt.dayofweek

        fig, axes = plt.subplots(5, 5, figsize=(22, 18))
        fig.suptitle("Promedio de transacciones por dia de la semana\n"
                    "5 terminales x 5 operaciones  |  "
                    "Magenta=dias habiles  Morado=fin de semana  "
                    "Barras de error=desviacion estandar",
                    fontsize=13, fontweight="bold", color=PURPLE)

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax    = axes[i, j]
                color = PALETTE[j % len(PALETTE)]
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                sub = daily[(daily["idTerminal"]==term) & (daily["oper"]==op)]
                dow_stats = (
                    sub.groupby("dow")["n_tx"]
                    .agg(["mean","std"])
                    .reindex(range(7), fill_value=0)
                    .reset_index()
                )

                ax.bar(dow_labels, dow_stats["mean"],
                    color=dow_colors, alpha=0.85,
                    width=0.65, edgecolor="white", linewidth=1.2)

                ax.errorbar(dow_labels, dow_stats["mean"],
                            yerr=dow_stats["std"],
                            fmt="none", color=PURPLE,
                            capsize=3, lw=1.2)

                prom = dow_stats["mean"].mean()
                ax.axhline(prom, color=self.colors["gray"],
                        lw=0.9, ls="--", alpha=0.8)

                mean_habil = dow_stats.loc[:4, "mean"].mean()
                mean_finde = dow_stats.loc[5:, "mean"].mean()
                if mean_habil > 0:
                    caida = (mean_habil - mean_finde) / mean_habil * 100
                    ax.text(0.97, 0.95, f"-{caida:.0f}%\nfinde",
                            ha="right", va="top",
                            transform=ax.transAxes,
                            fontsize=6.5, color=PURPLE, fontweight="bold")

                ax.set_title(f"T{term}/Op{op}", fontsize=8,
                            fontweight="bold", color=color)
                ax.set_ylabel("Tx promedio", fontsize=6)
                ax.tick_params(labelsize=6)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
                )

        fig.tight_layout()
        return self._save(fig, "fig15_dow_5x5.png")


    # ── 16. Promedio por hora del dia 5x5 ────────────────────────────────

    def plot_hora_5x5(
        self,
        df       : pd.DataFrame,
        terminals: list,
        opers    : list,
    ):
        """Fig 16 — Promedio de transacciones por hora del dia, grilla 5x5."""
        MAGENTA = self.colors["blue"]
        PURPLE  = self.colors["red"]
        PALETTE = [self.colors["blue"], self.colors["red"],
                self.colors["green"], self.colors["orange"],
                self.colors["gray"]]

        df_sel = df[
            df["idTerminal"].isin(terminals) &
            df["oper"].isin(opers)
        ].copy()
        n_dias = df_sel["fecha_dia"].nunique()

        def hora_color(h):
            if h < 6:  return self.colors["gray"]
            if h < 9:  return self.colors["orange"]
            if h < 13: return MAGENTA
            if h < 15: return self.colors["green"]
            if h < 19: return MAGENTA
            if h < 22: return self.colors["orange"]
            return self.colors["gray"]

        colores_hora = [hora_color(h) for h in range(24)]

        fig, axes = plt.subplots(5, 5, figsize=(24, 20))
        fig.suptitle("Promedio de transacciones por hora del dia\n"
                    "5 terminales x 5 operaciones  |  "
                    "Magenta=franja pico  Morado=fin semana  "
                    "Flecha=hora pico",
                    fontsize=13, fontweight="bold", color=PURPLE)

        for i, term in enumerate(terminals):
            for j, op in enumerate(opers):
                ax    = axes[i, j]
                color = PALETTE[j % len(PALETTE)]
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                sub = df_sel[
                    (df_sel["idTerminal"]==term) & (df_sel["oper"]==op)
                ]
                hora_stats = (
                    sub.groupby("hora")["oper"].count()
                    .reindex(range(24), fill_value=0)
                    .reset_index()
                    .rename(columns={"oper": "n_tx"})
                )
                hora_stats["media"] = hora_stats["n_tx"] / n_dias

                ax.bar(hora_stats["hora"], hora_stats["media"],
                    color=colores_hora, alpha=0.85,
                    width=0.85, edgecolor="white", linewidth=0.8)

                hora_pico = hora_stats.loc[hora_stats["media"].idxmax(), "hora"]
                val_pico  = hora_stats["media"].max()
                if val_pico > 0:
                    ax.annotate(
                        f"Pico\n{hora_pico}h",
                        xy=(hora_pico, val_pico),
                        xytext=(hora_pico + 2 if hora_pico < 20 else hora_pico - 4,
                                val_pico * 0.82),
                        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1),
                        fontsize=6, color=PURPLE, fontweight="bold"
                    )

                prom = hora_stats["media"].mean()
                ax.axhline(prom, color=self.colors["gray"],
                        lw=0.9, ls="--", alpha=0.8)

                ax.set_title(f"T{term}/Op{op}", fontsize=8,
                            fontweight="bold", color=color)
                ax.set_xlabel("Hora", fontsize=6)
                ax.set_ylabel("Tx promedio", fontsize=6)
                ax.set_xticks([0, 6, 12, 18, 23])
                ax.set_xticklabels(["0h","6h","12h","18h","23h"], fontsize=6)
                ax.tick_params(labelsize=6)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.1f}")
                )

        fig.tight_layout()
        return self._save(fig, "fig16_hora_5x5.png")
    
    # ── 17. Heatmap demanda predicha por operacion ────────────────────────

    def plot_demand_heatmap_oper(self, df_future: pd.DataFrame):
        """Fig 17 — Heatmap demanda predicha por operacion y dia."""
        pivot = (
            df_future.groupby(["oper", "ds"])["yhat"]
            .sum().reset_index()
            .pivot(index="oper", columns="ds", values="yhat")
        )
        pivot.columns = [
            pd.Timestamp(c).strftime("%d-%b") for c in pivot.columns
        ]
        fig, ax = plt.subplots(figsize=(16, 4))
        fig.suptitle(
            "Demanda total predicha por operacion (suma 5 terminales)\n"
            "2017-06-01 -> 2017-06-15  |  # Transacciones / dia",
            fontsize=12, fontweight="bold",
        )
        sns.heatmap(pivot, ax=ax, cmap="RdPu",
                    annot=True, fmt=".0f", linewidths=0.4,
                    cbar_kws={"label": "# Tx predichas", "shrink": 0.6})
        ax.set_xlabel("Dia")
        ax.set_ylabel("Operacion")
        fig.tight_layout()
        return self._save(fig, "fig17_demand_heatmap_oper.png")


    # ── 18. Heatmap demanda predicha terminal x operacion (total 15 dias) ─

    def plot_demand_heatmap_term_oper(self, df_future: pd.DataFrame):
        """Fig 18 — Heatmap demanda total predicha (15 dias) terminal x operacion."""
        pivot = (
            df_future.groupby(["idTerminal", "oper"])["yhat"]
            .sum().reset_index()
            .pivot(index="idTerminal", columns="oper", values="yhat")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            "Demanda total predicha — 15 dias\n"
            "Terminal x Operacion  |  # Transacciones totales",
            fontsize=12, fontweight="bold",
        )
        sns.heatmap(pivot, ax=ax, cmap="RdPu",
                    annot=True, fmt=".0f", linewidths=0.4,
                    cbar_kws={"label": "# Tx predichas", "shrink": 0.6})
        ax.set_xlabel("Operacion")
        ax.set_ylabel("Terminal ID")
        fig.tight_layout()
        return self._save(fig, "fig18_demand_heatmap_term_oper.png")


    # ── 19. Participacion porcentual por terminal y operacion ─────────────

    def plot_demand_participacion(self, df_future: pd.DataFrame):
        """Fig 19 — Participacion % de cada terminal y operacion en la demanda total."""
        MAGENTA = self.colors["blue"]
        PURPLE  = self.colors["red"]
        PALETTE = [self.colors["blue"], self.colors["red"],
                self.colors["green"], self.colors["orange"],
                self.colors["gray"]]

        total = df_future["yhat"].sum()

        # Por terminal
        by_term = (
            df_future.groupby("idTerminal")["yhat"]
            .sum()
            .sort_values(ascending=True)
        )
        pct_term = by_term / total * 100

        # Por operacion
        by_oper = (
            df_future.groupby("oper")["yhat"]
            .sum()
            .sort_values(ascending=True)
        )
        pct_oper = by_oper / total * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Participacion porcentual en la demanda predicha (15 dias)\n"
            "Izquierda: por terminal  |  Derecha: por operacion",
            fontsize=13, fontweight="bold", color=PURPLE
        )

        # Barras horizontales terminales
        ax = axes[0]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        bars = ax.barh(
            pct_term.index.astype(str),
            pct_term.values,
            color=MAGENTA, alpha=0.85,
            edgecolor="white", linewidth=1.2
        )
        for bar, val in zip(bars, pct_term.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold",
                    color=PURPLE)
        ax.set_xlabel("% del total predicho", fontsize=10)
        ax.set_ylabel("Terminal ID", fontsize=10)
        ax.set_title("Por terminal", fontsize=11, color=PURPLE)
        ax.set_xlim(0, pct_term.max() * 1.2)

        # Barras horizontales operaciones
        ax = axes[1]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        colors_oper = [PALETTE[i % len(PALETTE)]
                    for i in range(len(pct_oper))]
        bars = ax.barh(
            pct_oper.index.astype(str),
            pct_oper.values,
            color=colors_oper, alpha=0.85,
            edgecolor="white", linewidth=1.2
        )
        for bar, val in zip(bars, pct_oper.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold",
                    color=PURPLE)
        ax.set_xlabel("% del total predicho", fontsize=10)
        ax.set_ylabel("Operacion", fontsize=10)
        ax.set_title("Por operacion", fontsize=11, color=PURPLE)
        ax.set_xlim(0, pct_oper.max() * 1.2)

        fig.tight_layout()
        return self._save(fig, "fig19_demand_participacion.png")