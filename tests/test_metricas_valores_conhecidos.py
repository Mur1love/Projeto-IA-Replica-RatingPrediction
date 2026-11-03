"""Issue 10 — métricas com valores calculados à mão (labels 0..4, erro em estrelas)."""
import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    majority_class_label,
    predict_majority,
)


def test_predicao_perfeita():
    y = np.array([0, 1, 2, 3, 4])
    m = compute_metrics(y, y)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_pior_caso_1_estrela_contra_5_estrelas():
    m = compute_metrics(np.zeros(4, dtype=int), np.full(4, 4))
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["f1_macro"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(4.0)
    assert m["rmse"] == pytest.approx(4.0)


def test_valores_calculados_a_mao():
    # acertos: 3 de 4 -> accuracy 0.75 ; erro de 4 estrelas em 1 de 4
    # MAE = 4/4 = 1 ; RMSE = sqrt(16/4) = 2
    # F1 da classe 0 = 2/3 ; F1 da classe 4 = 0.8 ; macro = (2/3 + 0.8) / 2
    m = compute_metrics(np.array([0, 0, 4, 4]), np.array([0, 4, 4, 4]))
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(2.0)
    assert m["f1_macro"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_rmse_penaliza_erro_grande_mais_que_mae():
    # erros em estrelas: 0,0,0,0,4 -> MAE = 0.8 ; RMSE = sqrt(16/5)
    m = compute_metrics(np.zeros(5, dtype=int), np.array([0, 0, 0, 0, 4]))
    assert m["mae"] == pytest.approx(0.8)
    assert m["rmse"] == pytest.approx(math.sqrt(3.2))
    assert m["rmse"] > m["mae"]


def test_erro_e_simetrico_para_mais_e_para_menos():
    acima = compute_metrics(np.array([2]), np.array([4]))
    abaixo = compute_metrics(np.array([2]), np.array([0]))
    assert acima["mae"] == pytest.approx(abaixo["mae"])
    assert acima["rmse"] == pytest.approx(abaixo["rmse"])


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        ([0, 1, 2], [0, 1]),          # tamanhos diferentes
        ([0, 1, 5], [0, 1, 2]),       # label fora de 0..4
        ([0, 1, 2], [0, -1, 2]),      # label negativo
        ([0.0, np.nan], [0, 1]),      # NaN
        ([], []),                     # vazio
    ],
)
def test_entradas_invalidas_levantam_erro(y_true, y_pred):
    with pytest.raises(ValueError):
        compute_metrics(np.array(y_true), np.array(y_pred))


def test_classe_majoritaria_empate_vai_para_o_menor_label():
    assert majority_class_label(np.array([0, 0, 4, 4])) == 0


def test_baseline_majoritario_tem_accuracy_igual_a_frequencia_da_moda():
    y_treino = np.array([0, 1, 1, 2, 1])
    y_val = np.array([1, 1, 2, 0])
    previsao = predict_majority(y_treino, len(y_val))
    m = compute_metrics(y_val, previsao)
    assert m["accuracy"] == pytest.approx(0.5)  # 2 de 4 são a classe 1


def test_predict_majority_tamanho_negativo_e_rejeitado():
    with pytest.raises(ValueError):
        predict_majority(np.array([0, 1]), -1)
