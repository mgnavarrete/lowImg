import tkinter as tk
from tkinter import messagebox

class AntennaSelector:
    def __init__(self, root, antenas):
        self.root = root
        self.antenas = antenas
        self.current_index = 0
        self.iguales = []
        self.selected_indices = []
        self.create_widgets()
        self.update_interface()

    def create_widgets(self):
        self.root.title("Selector de Antenas")

        self.title_label = tk.Label(self.root, text="")
        self.title_label.pack(pady=10)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.buttons = []
        for i in range(len(self.antenas)):
            button = tk.Button(self.button_frame, text=str(i), bg='grey', 
                               command=lambda i=i: self.on_button_click(i))
            row = i // 20
            col = i % 20
            button.grid(row=row, column=col, padx=5, pady=5)
            self.buttons.append(button)

        self.next_button = tk.Button(self.root, text="Siguiente Antena", command=self.next_antenna)
        self.next_button.pack(pady=10)

    def on_button_click(self, index):
        button = self.buttons[index]
        current_color = button.cget('bg')
        new_color = 'green' if current_color == 'grey' else 'grey'
        button.config(bg=new_color)
        if new_color == 'green':
            self.selected_indices.append(index)
        else:
            self.selected_indices.remove(index)

    def next_antenna(self):
        self.save_selection()
        while self.current_index < len(self.antenas) - 1:
            self.current_index += 1
            if not any(self.current_index in sublist for sublist in self.iguales):
                break
        else:
            messagebox.showinfo("Info", "Has recorrido todos los elementos.")
            return

        self.update_interface()

    def save_selection(self):
        if self.selected_indices:
            self.iguales.append([self.current_index] + self.selected_indices)
            self.selected_indices = []

    def update_interface(self):
        self.title_label.config(text=f"Selecciona las antenas iguales a {self.current_index}")
        for button in self.buttons:
            button.config(bg='grey')

        if any(self.current_index in sublist for sublist in self.iguales):
            self.next_antenna()

def mainInterfaz(IDAntenas):
    root = tk.Tk()
    app = AntennaSelector(root, IDAntenas)
    root.mainloop()
    return app.iguales

if __name__ == "__main__":
    IDAntenas = [[0, 9, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [1, 3, 6, 1, 8, 11, 12, 6, 6, 0, 6, 5, 1, 4, None, None, None, None, None, None, None, None, None, None, None, None], [2, 7, 1, 4, 9, 8, 6, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [3, 6, 3, 7, 10, 7, 5, 10, 0, 3, 7, 13, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [4, 4, 8, 11, 1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [5, 8, 0, 3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [6, 1, 7, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [7, 5, 11, 5, 5, 1, 1, 4, 7, 12, 1, 10, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [9, 0, 10, 12, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, 2, 9, 0, 9, 7, 3, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, 4, 2, 7, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, 9, 0, 4, 5, 11, 8, 9, 1, 9, 2, 13, 10, 4, 11, 12, 10, None, None, None, None, None, None, None, None], [None, None, None, 6, 3, 0, 4, 11, 3, 2, 2, 3, 10, 3, 10, 4, 1, 12, 0, 6, 0, 4, None, None, None, None], [None, None, None, 8, 2, 6, 9, 9, 10, 9, 10, 12, 7, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, 6, 3, 10, 1, 5, 13, 0, 11, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, 2, 0, 7, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, 4, 2, 5, 2, 8, 4, 0, 9, 9, 8, 7, 0, 5, 6, 1, 9, 7, 0, 11, None, None], [None, None, None, None, None, 10, 3, 0, 1, 11, 5, 4, 3, 7, 7, 6, 6, 6, 12, 0, 4, 8, 9, 10, None, None], [None, None, None, None, None, None, 8, 2, 11, 7, 13, 7, 6, 5, 2, 1, 3, 2, 1, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, 8, 10, 11, 1, 0, 11, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, 4, 8, 8, 4, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, 5, 3, 9, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, 6, 12, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, 6, 5, 6, 9, 9, 13, 4, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, 11, 0, None, None, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, 12, 1, 11, 5, 10, 8, 5, 5, 7, 10, 7, 3, 4, 6], [None, None, None, None, None, None, None, None, None, None, None, None, None, 2, 1, 0, 4, 9, 11, 8, 2, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, None, 8, 0, 3, 9, 3, 8, 7, 11, 2, 4, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, 2, 2, 1, 2, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 6, 10, None, None, None, None, None, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 8, 7, 11, 10, 3, 1, 0, 3, 9, 6, 11], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 11, 7, 7, 9, 10, 1, 2, 6, 10, 2], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0, 4, 2, 8, 6, 8, 8, 8, 7], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, 10, 5, None, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 9, 4, 6, 9, 1, 7, 9, 8], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 13, 11, 3, 3, None, None, None, None], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 5, 6, 2, 11, 10], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 5, 4, 3, 0], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, 7, 1], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 5, 2, 3], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0, 9], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, 4], [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 5, 5]]
    newAntena = mainInterfaz(IDAntenas)
    finalAntenas = []
    for listAntenas in newAntena:
        allAntenas = []
        for idAntena in listAntenas:
            if allAntenas == []:
                allAntenas = IDAntenas[idAntena]
            else:
                for i in range(len(allAntenas)):
                    if allAntenas[i] == None:
                        allAntenas[i] = IDAntenas[idAntena][i]
        finalAntenas.append(allAntenas)
    
    print(finalAntenas)