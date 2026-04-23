import io
import pytest
import pandas as pd
import matplotlib.pyplot as plt

CSV_DATA = """\
label,llm_label
stratiform,stratiform
convective,convective
stratiform,convective
convective,stratiform
anvil,anvil
stratiform,stratiform
"""


@pytest.fixture
def sample_df():
    return pd.read_csv(io.StringIO(CSV_DATA))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_title_is_set(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    _, ax = plt.subplots()
    plot_confusion_matrix(sample_df, ax=ax)
    assert ax.get_title() == "Confusion Matrix"


def test_returns_none(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    _, ax = plt.subplots()
    result = plot_confusion_matrix(sample_df, ax=ax)
    assert result is None


def test_normalized(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    _, ax = plt.subplots()
    plot_confusion_matrix(sample_df, normalize="true", ax=ax)
    assert ax.get_title() == "Confusion Matrix"


def test_custom_column_names(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    df = sample_df.rename(columns={"label": "true", "llm_label": "pred"})
    _, ax = plt.subplots()
    plot_confusion_matrix(df, label_col="true", pred_col="pred", ax=ax)
    assert ax.get_title() == "Confusion Matrix"


def test_uses_gca_when_no_ax(sample_df):
    from lars.util.confusion_matrix import plot_confusion_matrix

    fig, ax = plt.subplots()
    plt.sca(ax)
    plot_confusion_matrix(sample_df)
    assert ax.get_title() == "Confusion Matrix"
