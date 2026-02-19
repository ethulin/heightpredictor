"""End-to-end tests for the Height Predictor web app."""
import re
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8888"


@pytest.fixture(autouse=True)
def goto_page(page: Page):
    page.goto(f"{BASE_URL}/docs/index.html")
    page.wait_for_load_state("networkidle")


def click_label(page: Page, for_id: str):
    """Click a label for a hidden radio input."""
    page.locator(f"label[for='{for_id}']").click()


# ── Page loads correctly ──────────────────────────────────────

def test_page_title(page: Page):
    expect(page).to_have_title("Adult Height Predictor")


def test_form_visible(page: Page):
    expect(page.locator("h1")).to_have_text("Height Predictor")
    expect(page.locator("#btn-predict")).to_be_visible()


# ── Sex toggle ────────────────────────────────────────────────

def test_sex_toggle(page: Page):
    expect(page.locator("#sex-m")).to_be_checked()
    click_label(page, "sex-f")
    expect(page.locator("#sex-f")).to_be_checked()
    expect(page.locator("#sex-m")).not_to_be_checked()


# ── Unit toggle converts input live ──────────────────────────

def test_unit_toggle_converts_input(page: Page):
    page.fill("#height-val", "110")
    click_label(page, "unit-in")
    val = float(page.input_value("#height-val"))
    assert abs(val - 43.3) < 0.2, f"Expected ~43.3 in, got {val}"
    # Toggle back
    click_label(page, "unit-cm")
    val2 = float(page.input_value("#height-val"))
    assert abs(val2 - 110.0) < 0.5, f"Expected ~110 cm, got {val2}"


# ── Prediction accuracy: Male 5yr 110cm ──────────────────────

def test_prediction_male_5yr_110cm(page: Page):
    """Reference: predicted=178.2, CI=[172.9, 183.6], SE=4.2, pct=59.4, r=0.81"""
    page.fill("#age-years", "5")
    page.fill("#age-months", "0")
    page.fill("#height-val", "110")
    page.click("#btn-predict")

    hero = page.locator("#hero-value").text_content()
    assert "178" in hero, f"Expected ~178.2 cm, got: {hero}"

    se_text = page.locator("#stat-se").text_content()
    assert "4" in se_text, f"Expected SE ~4.2, got: {se_text}"

    r_text = page.locator("#stat-r").text_content()
    assert r_text.startswith("0.81"), f"Expected r≈0.810, got: {r_text}"


# ── Prediction accuracy: Female 5yr 110cm ────────────────────

def test_prediction_female_5yr_110cm(page: Page):
    """Reference: predicted=165.5, CI=[159.6, 171.5], SE=4.6, pct=68.6, r=0.7"""
    click_label(page, "sex-f")
    page.fill("#age-years", "5")
    page.fill("#age-months", "0")
    page.fill("#height-val", "110")
    page.click("#btn-predict")

    hero = page.locator("#hero-value").text_content()
    assert "165" in hero or "166" in hero, f"Expected ~165.5 cm, got: {hero}"


# ── Prediction accuracy: Sena ────────────────────────────────

def test_prediction_sena_error(page: Page):
    """Age < 2 should show error."""
    click_label(page, "sex-f")
    page.fill("#age-years", "1")
    page.fill("#age-months", "0")
    page.fill("#height-val", "78")
    page.click("#btn-predict")

    error = page.locator("#error-msg")
    expect(error).to_be_visible()
    expect(error).to_contain_text("Age must be between 2 and 20")


# ── Prediction: Male 10yr 150cm (95% CI) ─────────────────────

def test_prediction_male_10yr_150cm(page: Page):
    """Reference: predicted=187.4, CI=[180.6, 194.2], SE=3.5, r=0.875"""
    page.fill("#age-years", "10")
    page.fill("#age-months", "0")
    page.fill("#height-val", "150")
    click_label(page, "conf-95")
    page.click("#btn-predict")

    hero = page.locator("#hero-value").text_content()
    assert "187" in hero or "188" in hero, f"Expected ~187.4 cm, got: {hero}"


# ── Prediction: Female 2yr6mo 90cm ───────────────────────────

def test_prediction_female_2yr6mo_90cm(page: Page):
    """Reference: predicted=163.4, CI=[155.5, 171.3], SE=4.8, r=0.669"""
    click_label(page, "sex-f")
    page.fill("#age-years", "2")
    page.fill("#age-months", "6")
    page.fill("#height-val", "90")
    click_label(page, "conf-90")
    page.click("#btn-predict")

    hero = page.locator("#hero-value").text_content()
    assert "163" in hero or "164" in hero, f"Expected ~163.4 cm, got: {hero}"


# ── Prediction: Male 18yr 175cm ──────────────────────────────

def test_prediction_male_18yr_175cm(page: Page):
    """Reference: predicted=175.7, CI=[173.9, 177.5], SE=1.4, r=0.98"""
    page.fill("#age-years", "18")
    page.fill("#age-months", "0")
    page.fill("#height-val", "175")
    page.click("#btn-predict")

    hero = page.locator("#hero-value").text_content()
    assert "175" in hero or "176" in hero, f"Expected ~175.7 cm, got: {hero}"

    r_text = page.locator("#stat-r").text_content()
    assert r_text.startswith("0.98"), f"Expected r≈0.980, got: {r_text}"


# ── Imperial output mode ─────────────────────────────────────

def test_imperial_output(page: Page):
    page.fill("#age-years", "5")
    page.fill("#age-months", "0")
    page.fill("#height-val", "110")
    page.click("#btn-predict")

    # Switch to imperial
    click_label(page, "unit-in")

    hero = page.locator("#hero-value").text_content()
    # 178.2 cm ≈ 5'10"
    assert "5\u2032" in hero or "6\u2032" in hero, f"Expected ft/in format, got: {hero}"


# ── Edge cases ───────────────────────────────────────────────

def test_edge_age_2(page: Page):
    page.fill("#age-years", "2")
    page.fill("#age-months", "0")
    page.fill("#height-val", "87")
    page.click("#btn-predict")

    results = page.locator("#results")
    expect(results).to_have_class(re.compile("show"))


def test_edge_age_20(page: Page):
    page.fill("#age-years", "20")
    page.fill("#age-months", "0")
    page.fill("#height-val", "175")
    page.click("#btn-predict")

    results = page.locator("#results")
    expect(results).to_have_class(re.compile("show"))


def test_results_hidden_initially(page: Page):
    results = page.locator("#results")
    expect(results).not_to_have_class(re.compile("show"))


# ── Screenshot tests ─────────────────────────────────────────

def test_screenshot_empty(page: Page):
    page.set_viewport_size({"width": 600, "height": 900})
    page.screenshot(path="tests/screenshots/empty_form.png", full_page=True)


def test_screenshot_with_results(page: Page):
    page.set_viewport_size({"width": 600, "height": 900})
    page.fill("#age-years", "5")
    page.fill("#age-months", "0")
    page.fill("#height-val", "110")
    page.click("#btn-predict")
    page.wait_for_timeout(500)
    page.screenshot(path="tests/screenshots/results_male.png", full_page=True)


def test_screenshot_female_results(page: Page):
    page.set_viewport_size({"width": 600, "height": 900})
    click_label(page, "sex-f")
    page.fill("#age-years", "10")
    page.fill("#age-months", "0")
    page.fill("#height-val", "140")
    page.click("#btn-predict")
    page.wait_for_timeout(500)
    page.screenshot(path="tests/screenshots/results_female.png", full_page=True)


def test_screenshot_mobile(page: Page):
    page.set_viewport_size({"width": 375, "height": 812})
    page.fill("#age-years", "8")
    page.fill("#age-months", "6")
    page.fill("#height-val", "130")
    page.click("#btn-predict")
    page.wait_for_timeout(500)
    page.screenshot(path="tests/screenshots/mobile_results.png", full_page=True)


def test_screenshot_imperial(page: Page):
    page.set_viewport_size({"width": 600, "height": 900})
    click_label(page, "unit-in")
    page.fill("#height-val", "43.3")
    page.fill("#age-years", "5")
    page.fill("#age-months", "0")
    page.click("#btn-predict")
    page.wait_for_timeout(500)
    page.screenshot(path="tests/screenshots/imperial_results.png", full_page=True)
