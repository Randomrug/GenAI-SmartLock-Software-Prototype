from .sms import send_sms
from .call import make_call

print("\nSMART LOCK ALERT SYSTEM (TEST)")
print("1 -> 2 or 3 denials (Warning)")
print("2 -> System Locked (Emergency)")

choice = input("Enter choice (1/2): ")

if choice == "1":
    send_sms()
elif choice == "2":
    make_call()
else:
    print("[ERROR] Invalid choice.")
