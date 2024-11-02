import cv2
import numpy as np
import time

# Variables globales para el manejo de eventos del mouse
cuadrados = []
cuadrado_seleccionado = None
offset_x = 0
offset_y = 0
ultimo_tablero = None  # Para almacenar el estado del tablero anterior
turno_robot = True  # True si es el turno de tu robot, False si es el turno del oponente
ultimo_tiempo_turno = time.time()
tablero = np.zeros((6, 5), dtype=int)

# Crear una tabla de corrección gamma
def ajustar_gamma(imagen, gamma=1.2):
    tabla = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(imagen, tabla)

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

def detectar_fichas_webcam(num_filas=6, num_columnas=5, gamma=1.5, intervalo=5, tamano_celda=50):
    global cuadrados, ultimo_tablero, turno_robot, ultimo_tiempo_turno, tablero

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

        #

        # Dibuja los cuadrados en sus posiciones actuales
        for idx, cuadrado in enumerate(cuadrados):
            x_inicio, y_inicio, tamano_celda = cuadrado
            x_fin = x_inicio + tamano_celda
            y_fin = y_inicio + tamano_celda
            cv2.rectangle(imagen_procesada, (int(x_inicio), int(y_inicio)), (int(x_fin), int(y_fin)), (255, 0, 0), 1)

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
           
            if ultimo_tablero is not None and np.array_equal(tablero, ultimo_tablero):
                if not turno_robot and time.time() - ultimo_tiempo_turno >= 40:
                    print("El robot oponente no hizo su jugada. Turno perdido.")
                    turno_robot = True
                    ultimo_tiempo_turno = time.time()
            else:
                turno_robot = not turno_robot
                ultimo_tiempo_turno = time.time()
                ultimo_tablero = tablero.copy()
                return tablero
                
            # Imprimir el estado del tablero en la consola
            print("IMPRIENDO DE COMPUTER VISION")
            print("Estado del tablero (1 = ficha presente, 0 = vacío):")
            print(tablero)

        # Mostrar la imagen procesada con la cuadrícula y detecciones
        cv2.imshow("Detección de fichas en el tablero", imagen_procesada)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()
    

# Llamar a la función para iniciar la detección con cuadrados más pequeños
detectar_fichas_webcam(tamano_celda=40)