import requests

def main():
    print("Добро пожаловать! Чтобы выйти, введите пустую строку или Ctrl+C.")
    while True:
        question = input("\nВведите ваш вопрос: ").strip()
        if not question:
            print("Выход.")
            break
        
        resp = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question},
            timeout=30
        )
        
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "Нет поля answer в ответе.")
            print("Ответ:", answer)
        else:
            print("Ошибка:", resp.status_code, resp.text)

if __name__ == "__main__":
    main()
