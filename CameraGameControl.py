import cv2 
import numpy as np
import time
from ConnectFourAI import bestMove, AI_PLAYER, OTHER_PLAYER


# Configuración inicial y Variables globales para el manejo de eventos del mouse
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 6
cuadrados = []
cuadrado_seleccionado = None
offset_x = 0
offset_y = 0
turno_robot = True  # True si es el turno de tu robot, False si es el turno del oponente
ultimo_tiempo_turno = time.time()
tablero = np.zeros((6, 5), dtype=int)
calibration_time = True
move_detected = False
global best_move

# Matrices para el estado del tablero
board_detected = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X), dtype=int)  # 0s y 1s (detección básica)
board_game = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X), dtype=int)  # 2s y 3s (fichas del robot y oponente)


# Crear una tabla de corrección gamma
def ajustar_gamma(imagen, gamma=1.2):
    tabla = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(imagen, tabla)


#Movimiento de Cuadrados por Mouse
def mouse_event(event, x, y, flags, param):
    global cuadrados, cuadrado_seleccionado, offset_x, offset_y


    if event == cv2.EVENT_LBUTTONDOWN:
        # Cuando se presiona el botón izquierdo del mouse
        for idx, cuadrado in enumerate(cuadrados):
            x_inicio, y_inicio, tamano_celda = cuadrado
            x_fin = x_inicio + tamano_celda
            y_fin = y_inicio + tamano_celda
            if x_inicio <= x <= x_fin and y_inicio <= y <= y_fin:
                cuadrado_seleccionado = idx
                offset_x = x - x_inicio
                offset_y = y - y_inicio
                break


    elif event == cv2.EVENT_MOUSEMOVE:
        # Cuando se mueve el mouse
        if cuadrado_seleccionado is not None:
            # Actualizar la posición del cuadrado seleccionado
            x_nuevo = x - offset_x
            y_nuevo = y - offset_y
            tamano_celda = cuadrados[cuadrado_seleccionado][2]
            cuadrados[cuadrado_seleccionado] = (x_nuevo, y_nuevo, tamano_celda)


    elif event == cv2.EVENT_LBUTTONUP:
        # Cuando se suelta el botón izquierdo del mouse
        cuadrado_seleccionado = None

def detectar_fichas_webcam(num_filas=BOARD_SIZE_Y, num_columnas=BOARD_SIZE_X, gamma=1.5, intervalo=5, tamano_celda=50):
    global cuadrados, turno_robot, ultimo_tiempo_turno, tablero, calibration_time, move_detected

    # Iniciar la captura de la webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la webcam.")
        return None

    # Variables para el temporizador
    ultimo_tiempo = time.time()

    # Crear los cuadrados iniciales distribuidos en una grilla
    cuadrados = []
    for fila in range(num_filas):
        for columna in range(num_columnas):
            x_inicio = 100 + columna * (tamano_celda + 10)
            y_inicio = 100 + fila * (tamano_celda + 10)
            cuadrados.append([x_inicio, y_inicio, tamano_celda])

    # Crear una ventana y asignarle la función de callback del mouse
    cv2.namedWindow("Detección de fichas en el tablero")
    cv2.setMouseCallback("Detección de fichas en el tablero", mouse_event)

    while True:
        ret, imagen = cap.read()
        if not ret:
            print("Error: No se pudo capturar imagen de la webcam.")
            break

        # Convertir a escala de grises y redimensionar para una visualización uniforme
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen_gris = cv2.resize(imagen_gris, (800, 600))

        # Aplicar corrección gamma y desenfoque
        imagen_gamma = ajustar_gamma(imagen_gris, gamma=gamma)
        imagen_blur = cv2.GaussianBlur(imagen_gamma, (5, 5), 0)

        # Crear filtro de paso alto (opcional)
        imagen_procesada = cv2.addWeighted(imagen_gamma, 1.5, imagen_blur, -0.5, 0)

        # Dibuja los cuadrados en sus posiciones actuales
        for idx, cuadrado in enumerate(cuadrados):
            x_inicio, y_inicio, tamano_celda = cuadrado
            x_fin = x_inicio + tamano_celda
            y_fin = y_inicio + tamano_celda
            cv2.rectangle(imagen_procesada, (int(x_inicio), int(y_inicio)), (int(x_fin), int(y_fin)), (255, 0, 0), 1)
        
        if calibration_time:
            print("Tiempo de calibración")
            if time.time() - ultimo_tiempo >= 60:
                calibration_time = False
        else:
            # Actualizar cada 'intervalo' segundos
            if time.time() - ultimo_tiempo >= intervalo:
                ultimo_tiempo = time.time()  # Reiniciar el temporizador

                # Revisar cada celda para detectar ocupación basada en tonos grises
                for idx, cuadrado in enumerate(cuadrados):
                    x_inicio, y_inicio, tamano_celda = cuadrado
                    x_fin = x_inicio + tamano_celda
                    y_fin = y_inicio + tamano_celda

                    # Asegurarse de que los índices estén dentro de los límites de la imagen
                    x_inicio = int(max(x_inicio, 0))
                    y_inicio = int(max(y_inicio, 0))
                    x_fin = int(min(x_fin, imagen_gamma.shape[1]))
                    y_fin = int(min(y_fin, imagen_gamma.shape[0]))

                    # Extraer la región de interés (ROI) para cada celda
                    celda = imagen_gamma[y_inicio:y_fin, x_inicio:x_fin]

                    # Calcular el valor promedio de gris en la celda
                    if celda.size > 0:
                        valor_medio = np.mean(celda)
                    else:
                        valor_medio = 255  # Si la celda está fuera de la imagen, asignar un valor alto

                    # Definir un umbral para detectar fichas
                    if valor_medio < 80:  # Ajusta este valor según tus necesidades
                        tablero[idx // num_columnas, idx % num_columnas] = 1
                        # Dibujar un rectángulo verde en las celdas detectadas como ocupadas
                        cv2.rectangle(imagen_procesada, (int(x_inicio), int(y_inicio)), (int(x_fin), int(y_fin)), (0, 255, 0), 2)
                    else:
                        tablero[idx // num_columnas, idx % num_columnas] = 0  # Marcar como vacío si no cumple el umbral
                
                #Gestionar cambio de tablero y turnos

                # Imprimir el estado del tablero en la consola
                print("IMPRIENDO DE COMPUTER VISION")
                print("Estado del tablero (1 = ficha presente, 0 = vacío):")
                board_detected = tablero
                print(board_detected)

        # Mostrar la imagen procesada con la cuadrícula y detecciones
        cv2.imshow("Detección de fichas en el tablero", imagen_procesada)


        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Liberar la cámara y cerrar las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()


# Función para detectar cambios en el tablero
def detect_changes():
    global turno_robot
    for row in range(BOARD_SIZE_Y):
        for col in range(BOARD_SIZE_X):
            if board_detected[row][col] == 1 and board_game[row][col] == 0:
                # Movimiento detectado de oponente
                board_game[row][col] = OTHER_PLAYER
                print("Movimiento detectado del oponente.")
                print("IMPRIENDO DE TESTING")
                print("Estado del tablero:")
                print(board_game)
                turno_robot = True
                return
            elif board_detected[row][col] == 0 and board_game[row][col] == AI_PLAYER:
                # Verificación si el robot falló en situar la ficha
                print("Error: El robot no colocó la ficha en la posición indicada.")
                turno_robot = True
                return


# Función para ejecutar el movimiento del robot
def robot_play():
    move_col = bestMove(board_game, AI_PLAYER, OTHER_PLAYER)
    for row in range(BOARD_SIZE_Y - 1, -1, -1):
        if board_game[row][move_col] == 0:
            board_game[row][move_col] = AI_PLAYER
            break
    print(f"Robot coloca ficha en columna: {move_col}")
    print("IMPRIENDO DE TESTING COLUMNA CON JUGADA")
    print("Estado del tablero:")
    print(board_game)
    

detectar_fichas_webcam()