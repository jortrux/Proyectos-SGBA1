import os
import time
import datetime
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

download_dir_temp = "downloads_temp"
download_dir_potencia = "downloads_potencia"
download_dir_gen_consumo = "downloads_generacion_consumo"
download_dir_gen_co2 = "downloads_generacion_co2"

for d in [download_dir_temp, download_dir_potencia, download_dir_gen_consumo, download_dir_gen_co2]:
    os.makedirs(d, exist_ok=True)

chrome_options = Options()

prefs = {
    "download.default_directory": os.path.abspath(download_dir_temp),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 40)  # Espera hasta 40 seg

def wait_for_new_file(before_files, folder, timeout=30):
    end_time = time.time() + timeout
    while time.time() < end_time:
        current_files = set(os.listdir(folder))
        new_files = current_files - before_files
        if new_files:
            return new_files.pop()
        time.sleep(0.5)
    return None

## Solucion de gpt para el  problema de error en descarga
def click_retry_in_downloads():
    driver.execute_script("window.open('chrome://downloads');")
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(1)  # Espera breve para que cargue la página

    try:
        retry_button = driver.execute_script("""
            return document.querySelector('downloads-manager')
                .shadowRoot.querySelector('downloads-item')
                .shadowRoot.querySelector('#retryButton');
        """)
        if retry_button:
            driver.execute_script("arguments[0].click();", retry_button)
            print("Se pulsó el botón Retry en chrome://downloads.")
        else:
            print("No se encontró el botón Retry en chrome://downloads.")
    except Exception as e:
        print("Error al intentar pulsar Retry:", e)
    ## por el retry del error de descarga
    time.sleep(1)  
    driver.close()
    driver.switch_to.window(driver.window_handles[0])

def download_csv_by_h1_text(h1_text, date_str, name_prefix, final_folder):
    """
    Busca un <h1> con 'h1_text', ubica el contenedor widget,
    hace clic en 'Compartir' -> 'CSV' y mueve el archivo a la carpeta final.
    Si la descarga falla, se reintenta pulsando 'Retry' en chrome://downloads.
    """
    try:
        h1_element = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, f"//h1[contains(normalize-space(), '{h1_text}')]")
            )
        )
    except:
        print(f"  - [{name_prefix}] No se encontró <h1> con texto '{h1_text}'.")
        return

    try:
        widget = h1_element.find_element(By.XPATH, "./ancestor::div[contains(@class,'widget')]")
    except:
        print(f"  - [{name_prefix}] No se encontró el <div class='widget'> para '{h1_text}'.")
        return

    try:
        share_btn = widget.find_element(By.CSS_SELECTOR, ".toolbar-share")
        share_btn.click()
        time.sleep(0.5)
    except:
        print(f"  - [{name_prefix}] Botón 'Compartir' no encontrado en widget '{h1_text}'.")
        return

    try:
        csv_link = widget.find_element(By.ID, "mydivcsv")
    except:
        print(f"  - [{name_prefix}] Opción CSV no encontrada en widget '{h1_text}'.")
        return

    before_files = set(os.listdir(download_dir_temp))
    csv_link.click()

    new_file = wait_for_new_file(before_files, download_dir_temp, timeout=40)
    
    if not new_file:
        print(f"  - [{name_prefix}] No se detectó el archivo, intentando reintentar descarga...")
        click_retry_in_downloads()
        new_file = wait_for_new_file(before_files, download_dir_temp, timeout=40)
    
    if new_file:
        old_path = os.path.join(download_dir_temp, new_file)
        new_filename = f"{name_prefix}_{date_str}.csv"
        new_path = os.path.join(final_folder, new_filename)
        shutil.move(old_path, new_path)
        print(f"  - [{name_prefix}] CSV descargado y renombrado a: {new_path}")
    else:
        print(f"  - [{name_prefix}] Aún no se detectó el archivo tras el reintento.")


start_date = datetime.date(2017, 1, 4)
end_date = datetime.date(2020, 1, 1)

try:
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%d-%m-%Y")
        url = f"https://www.esios.ree.es/es/generacion-y-consumo?date={date_str}"
        print(f"\n[INFO] Abriendo página para fecha: {date_str}")
        driver.get(url)
        time.sleep(6)

        # Cerrar cookies, si aparecen
        # try:
        #     accept_cookies_btn = wait.until(
        #         EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Aceptar')]"))
        #     )
        #     accept_cookies_btn.click()
        # except:
        #     pass

        # 1) Potencia (h1 => "Potencia")
        download_csv_by_h1_text("Potencia", date_str, "potencia", download_dir_potencia)

        # 2) Generación y consumo (h1 => "Generación y consumo")
        download_csv_by_h1_text("Generación y consumo", date_str, "generacion_consumo", download_dir_gen_consumo)

        # 3) Generación libre de CO₂ (h1 => "Generación libre de CO", por si hay un caracter especial)
        #    Podrías usar: "Generación libre de CO₂"
        #    Si no funciona con el caracter especial, usa un contains más parcial:
        download_csv_by_h1_text("Generación libre de CO", date_str, "generacion_libre_co2", download_dir_gen_co2)

        current_date += datetime.timedelta(days=1)

finally:
    driver.quit()

print("\n[INFO] Proceso finalizado.")
