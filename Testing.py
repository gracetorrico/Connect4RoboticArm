import time 
import numpy as np
from ConnectFourAI import bestMove, AI_PLAYER, OTHER_PLAYER

# Configuración inicial
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 6
robot_turn = True  

# Matrices para el estado del tablero
board_detected = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X), dtype=int)  # 0s y 1s (detección básica)
board_game = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X), dtype=int)  # 2s y 3s (fichas del robot y oponente)

# Función para actualizar la matriz del tablero
def update_board():
    print("Por favor ingresa la matriz del tablero:")
    global board_detected
    board_input = input("Ingresa la matriz del tablero: ")
    board_detected = np.array(eval(board_input), dtype=int)
    detect_changes()

# Función para detectar cambios en el tablero
def detect_changes():
    global robot_turn
    for row in range(BOARD_SIZE_Y):
        for col in range(BOARD_SIZE_X):
            if board_detected[row][col] == 1 and board_game[row][col] == 0:
                # Movimiento detectado de oponente
                board_game[row][col] = OTHER_PLAYER
                print("Movimiento detectado del oponente.")
                robot_turn = True
                return
            elif board_detected[row][col] == 0 and board_game[row][col] == AI_PLAYER:
                # Verificación si el robot falló en situar la ficha
                print("Error: El robot no colocó la ficha en la posición indicada.")
                robot_turn = True
                return

# Función para ejecutar el movimiento del robot
def robot_play():
    move_col = bestMove(board_game, AI_PLAYER, OTHER_PLAYER)
    for row in range(BOARD_SIZE_Y - 1, -1, -1):
        if board_game[row][move_col] == 0:
            board_game[row][move_col] = AI_PLAYER
            break
    print(f"Robot coloca ficha en columna: {move_col}")
    print("Estado del tablero:")
    print(board_game)

# Ciclo principal del juego
while True:
    if robot_turn:
        print("Es el turno del robot.")
        robot_play()
        robot_turn = False
        start_time = time.time()
    else:
        print("Es el turno del oponente.")
        start_time = time.time()
        while time.time() - start_time < 40:
            time.sleep(5)
            print("Esperando actualización del tablero...")
            update_board()
            if robot_turn:
                break
        else:
            print("Tiempo de espera terminado, el robot toma su turno.")
            robot_turn = True
