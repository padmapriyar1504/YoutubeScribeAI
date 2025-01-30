
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service
from selenium.common.exceptions import TimeoutException

# Specify the Edge WebDriver path
edge_driver_path = "C:\\Users\\ADMIN\\Desktop\\devomlo - Copy\\edgedriver_win64\\msedgedriver.exe"
service = Service(edge_driver_path)
driver = webdriver.Edge(service=service)

try:
    # Open the web application
    driver.get("http://127.0.0.1:5000")
    driver.maximize_window()

    # Wait for the input field to be visible and enter the YouTube URL
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.ID, "url"))
    ).send_keys("https://www.youtube.com/watch?v=JzPfMbG1vrE")

    # Click the "Start Process" button
    start_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
    )
    start_button.click()

    # Wait until the summary is no longer "Awaiting content..."
    WebDriverWait(driver, 120).until(
        lambda d: d.find_element(By.CSS_SELECTOR, "#summary p").text != "Awaiting content..."
    )

    # Retrieve and print the summary
    summary_text = driver.find_element(By.CSS_SELECTOR, "#summary p").text
    print("Success!")
    print("Summary:", summary_text)

except TimeoutException:
    print("Test failed: Timed out while waiting for elements or summary generation.")

finally:
    # Close the browser immediately
    driver.quit()
    print("Success!")