from choice_functions import *

if __name__ == '__main__':
    print("\n\nINSERIRE UNA SCELTA TRA:")
    print("--------------------------------------")
    print("1 - Preprocessing \n")
    print("2 - Visualizzazione dei grafici sui dati\n")
    print("3 - Training, Testing e confronto dei modelli considerati \n")
    print("4 - Esecuzione completa del programma\n")

    # Flag per la scelta della parte di codice da eseguire:

    # choice = int(input("\nValore da inserire: "))
    choice = 3

    if choice == 1:  # Preprocessing
        preprocess_choice()
    elif choice == 2:  # Visualizzazione
        visualization_choice()
    elif choice == 3:  # Training, Testing e confronto dei modelli considerati
        model_management()
    elif choice == 4:  # Esecuzione completa del programma
        preprocess_choice()
        visualization_choice()
        model_management()
