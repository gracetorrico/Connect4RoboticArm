import cv2 
import numpy as np
import time
import serial 
from ConnectFourAI import bestMove, AI_PLAYER, OTHER_PLAYER, checkWin


# Configuración inicial y Variables globales para el manejo de eventos del mouse
BOARD_SIZE_X = 5
BOARD_SIZE_Y = 6
cuadrados = []
cuadrado_seleccionado = None
offset_x = 0
offset_y = 0
#ser = serial.Serial('COM5', 9600, timeout=1)

# Matrices para el estado del tablero
board_detected = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X), dtype=int)  # 0s y 1s (detección básica)
board_game = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X), dtype=int)  # 2s y 3s (fichas del robot y oponente)

def send_state(state):
    if 1 <= state <= 5:
        # Convert state to byte and send over UART
        #ser.write(bytes([state]))
        print(f"Sent state: {state}")
    else:
        print("State out of range (1-5)")

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

def detectar_fichas_webcam(num_filas=6, num_columnas=5, gamma=1.5, intervalo=2, tamano_celda=50):
    global cuadrados

    # Iniciar la captura de la webcam
    cap = cv2.VideoCapture(1)  # Cambiar a 0 si la cámara 1 no funciona
    if not cap.isOpened():
        print("Error: No se pudo acceder a la webcam.")
        return None

    # Variables para el temporizador
    check_tiempo = time.time()

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

    # Variables para el periodo de calibración
    calibration_start_time = time.time()
    calibration_duration = 70  # Reducido a 10 segundos para pruebas
    calibration_done = False
    empty_board_state = None

    # Variables para el juego
    per_turn_timeout = 45  # segundos
    turn_start_time = None
    current_player = AI_PLAYER  # Nuestro robot empieza
    last_board_state = None  # Inicializar como None
    pending_changes = {}
    game_over = False

    # Variable para almacenar la recomendación de la IA
    ai_recommendation = None

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
            color = (255, 0, 0)
            # Resaltar la columna recomendada para el robot
            if current_player == AI_PLAYER and ai_recommendation is not None:
                col_recomendada = ai_recommendation
                if idx % num_columnas == col_recomendada:
                    color = (0, 255, 255)  # Amarillo para la recomendación
            cv2.rectangle(imagen_procesada, (int(x_inicio), int(y_inicio)), (int(x_fin), int(y_fin)), color, 1)

        # Dibuja las fichas confirmadas en sus posiciones
        if last_board_state is not None:
            for fila in range(num_filas):
                for columna in range(num_columnas):
                    jugador = last_board_state[fila, columna]
                    if jugador != 0:
                        idx_cuadrado = fila * num_columnas + columna
                        x_inicio, y_inicio, tamano_celda = cuadrados[idx_cuadrado]
                        x_fin = x_inicio + tamano_celda
                        y_fin = y_inicio + tamano_celda
                        color_jugador = (0, 0, 255) if jugador == AI_PLAYER else (0, 255, 0)
                        cv2.rectangle(imagen_procesada, (int(x_inicio), int(y_inicio)), (int(x_fin), int(y_fin)), color_jugador, -1)

        # Periodo de calibración
        if not calibration_done:
            tiempo_calibracion = time.time() - calibration_start_time
            cv2.putText(imagen_procesada, f"Calibrando... {int(calibration_duration - tiempo_calibracion)} s restantes", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            if tiempo_calibracion >= calibration_duration:
                calibration_done = True
                # Capturar el estado inicial del tablero (vacío)
                empty_board_state = np.zeros((num_filas, num_columnas), dtype=int)
                last_board_state = np.zeros((num_filas, num_columnas), dtype=int)
                turn_start_time = time.time()
            else:
                # Mostrar la imagen procesada durante la calibración
                cv2.imshow("Detección de fichas en el tablero", imagen_procesada)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue  # Ir a la siguiente iteración del bucle

        # Si el juego ha terminado
        if game_over:
            cv2.putText(imagen_procesada, "Juego terminado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Detección de fichas en el tablero", imagen_procesada)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Variables de tiempo para el turno actual
        if turn_start_time is None:
            turn_start_time = time.time()

            # Si es el turno del jugador, obtener la recomendación
            if current_player == AI_PLAYER:
                gameState = last_board_state.copy()
                ai_recommendation = bestMove(gameState.tolist(), AI_PLAYER, OTHER_PLAYER)
                if ai_recommendation is not None:
                    #send_state(ai_recommendation)
                    print(f"La IA recomienda colocar la ficha en la columna: {ai_recommendation}")
                else:
                    print("No hay movimientos posibles. El juego ha terminado en empate.")
                    game_over = True
                    continue

        tiempo_transcurrido_turno = time.time() - turn_start_time
        tiempo_restante_turno = per_turn_timeout - tiempo_transcurrido_turno

        cv2.putText(imagen_procesada, f"Jugador {current_player} - Tiempo restante: {int(tiempo_restante_turno)} s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar la recomendación en la pantalla
        if current_player == AI_PLAYER and ai_recommendation is not None:
            cv2.putText(imagen_procesada, f"Recomendación IA: Columna {ai_recommendation}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Actualizar cada 'intervalo' segundos
        if time.time() - check_tiempo >= intervalo:
            check_tiempo = time.time()  # Reiniciar el temporizador

            # Crear una matriz para representar el estado del tablero
            tablero_actual = np.zeros((num_filas, num_columnas), dtype=int)

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
                if valor_medio < 130:  # Ajusta este valor según tus necesidades
                    tablero_actual[idx // num_columnas, idx % num_columnas] = 1
                else:
                    tablero_actual[idx // num_columnas, idx % num_columnas] = 0  # Marcar como vacío si no cumple el umbral

            # Comparar el estado actual con el anterior para detectar cambios
            if last_board_state is not None:
                # Detectar donde tablero_actual es 1 y last_board_state es 0
                diferencia = (tablero_actual == 1) & (last_board_state == 0)
                posibles_nuevas_fichas = np.argwhere(diferencia)

                for celda in posibles_nuevas_fichas:
                    fila, columna = celda
                    clave_celda = (fila, columna)

                    if clave_celda not in pending_changes:
                        # Primera detección, agregar a pendientes
                        pending_changes[clave_celda] = 1
                        print(f"Cambio detectado en celda: {celda}")
                    else:
                        # Segunda detección consecutiva, confirmar nueva ficha
                        pending_changes[clave_celda] += 1
                        if pending_changes[clave_celda] >= 2:
                            print("Confirmación de cambio!")
                            # Confirmar la ficha para el jugador actual
                            last_board_state[fila, columna] = current_player
                            print(f"Jugador {current_player} ha colocado una ficha en ({fila}, {columna})")

                            # Comprobar si el jugador actual ha ganado
                            winner = checkWin(last_board_state.tolist())
                            if winner == current_player:
                                if current_player == AI_PLAYER:
                                    print("¡Jugador 1 ha ganado el juego!")
                                else:
                                    print("¡Jugador 2 ha ganado el juego!")
                                game_over = True

                            # Cambiar al siguiente jugador
                            current_player = AI_PLAYER if current_player == OTHER_PLAYER else OTHER_PLAYER
                            turn_start_time = None
                            pending_changes = {}
                            ai_recommendation = None  # Reiniciar la recomendación
                            break  # Salir del bucle de detección de fichas

                # Limpiar las celdas que no han cambiado
                for clave_celda in list(pending_changes.keys()):
                    fila, columna = clave_celda
                    if tablero_actual[fila, columna] == last_board_state[fila, columna]:
                        del pending_changes[clave_celda]

            else:
                # Si es la primera iteración, inicializar last_board_state
                last_board_state = np.zeros((num_filas, num_columnas), dtype=int)
                print("Tablero:")
                print(last_board_state)

        # Si se acabó el tiempo del turno sin que el jugador coloque una ficha
        if tiempo_restante_turno <= 0:
            print(f"Jugador {current_player} ha perdido su turno por tiempo.")
            # Cambiar al siguiente jugador
            current_player = AI_PLAYER if current_player == OTHER_PLAYER else OTHER_PLAYER
            turn_start_time = None
            pending_changes = {}
            ai_recommendation = None  # Reiniciar la recomendación

        # Mostrar la imagen procesada con la cuadrícula y detecciones
        cv2.imshow("Detección de fichas en el tablero", imagen_procesada)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas de OpenCV
    #ser.close()
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para iniciar la detección con cuadrados más pequeños
detectar_fichas_webcam(tamano_celda=40)
