# Cambios de organizacion en mmdetection — 30 de agosto de 2026

Resumen de que se movio, que se elimino y por que. Nada de codigo fue modificado:
solo se reubicaron carpetas y se consolido en un unico lugar.

## Situacion anterior: tenias DOS copias

| Ubicacion | Que era |
|---|---|
| `C:\workspace\mmdetection` | Tu fork `Ivan1721/mmdetection` con tus commits y `work_dirs` |
| `OneDrive\...\vision\Transformer\mmdetection` | Clon de `open-mmlab` a secas, sin tu fork configurado |

La segunda no tenia tu remoto, asi que no podias pushear desde ahi. Era una copia
huerfana, probablemente anterior o paralela.

## Que se hizo

### 1. El repo se movio fuera de OneDrive
    C:\workspace\mmdetection   ->   C:\Users\garci\repos\mmdetection

Motivo: OneDrive sincroniza los objetos de `.git` mientras git los reescribe, lo que
puede corromper el indice del repositorio. Ademas duplicaba en la nube algo que
GitHub ya versiona.

Se dejo un acceso directo en `OneDrive\Desktop\GitHub\mmdetection.lnk` para que siga
siendo alcanzable desde tu estructura de carpetas.

`C:\workspace` quedo vacia y se elimino.

### 2. Se elimino el clon secundario
Se borro `vision\Transformer\mmdetection` (3.734 archivos, 0.34 GB): era el clon de
open-mmlab sin modificar, mas archivos `__pycache__`.

Antes de borrarlo se rescataron 5 archivos que no eran del clon. Al compararlos
contra este fork resulto que:

| Archivo | Resultado |
|---|---|
| `solov2_r50_fpn_1x_coco.py` | IDENTICO, ya estaba commiteado aqui |
| `solov2_..._a357fa23.pth` (178 MB) | IDENTICO, ya estaba aqui |
| `demo/out/preds/demo.json` | difiere 8 bytes (misma inferencia, otra corrida) |
| `demo/out/vis/demo.jpg` | difiere 417 bytes (idem) |
| `rtmdet_tiny_8xb32-300e_coco.py` | UNICO — no estaba en el fork |

### 3. Se recupero el config de rtmdet
`rtmdet_tiny_8xb32-300e_coco.py` se copio a la raiz de este repo y se commiteo:

    commit 4b99c029  "Add rtmdet_tiny config recovered from secondary clone"
    1 file changed, 547 insertions(+)

PENDIENTE: ese commit todavia NO esta pusheado. Corre `git push` cuando quieras.

### 4. Se elimino la carpeta de rescate
`vision\Transformer\mmdetection_mis_archivos` (5 archivos, 178 MB) se borro tras
confirmar que todo su contenido util ya estaba en este repo.

## Estado actual

    C:\Users\garci\repos\mmdetection      <- unica copia, fork Ivan1721, rama main
      +- work_dirs\                       <- 5.03 GB de entrenamientos, en .gitignore
      +- CAMBIOS_ORGANIZACION_2026-08-30.md  (este archivo)

    OneDrive\...\vision\Transformer\
      +- codigos\        tu codigo
      +- Manzana\        dataset COCO con anotaciones
      +- COMO_RESTAURAR.txt

## Sobre work_dirs (5.03 GB, sin respaldo en ningun lado)

Estan en `.gitignore`, o sea que NO viajan a GitHub. Contenido:

| Que | Peso | mAP |
|---|---|---|
| `mask2former_apples/iter_2000.pth` | 505 MB | **0.975** (el mejor) |
| `mask_rcnn_apples_instance/epoch_12.pth` | 335 MB | 0.612 |
| `mask_rcnn_apples_instance/epoch_1..11.pth` | 3.68 GB | intermedios, descartables |
| `mask2former_apples/iter_200.pth` | 504 MB | arranque de prueba |
| logs, configs y `scalars.json` | ~1 MB | curvas de entrenamiento — CONSERVAR |

Conservando solo lo util serian ~840 MB en vez de 5.03 GB.

NOTA: el archivo `mask_rcnn_apples_instance/last_checkpoint` apunta a
`work_dirs\mask2former_apples_instance\epoch_12.pth`, ruta que no existe — parece que
la carpeta se renombro en algun momento. El checkpoint si existe, en
`mask_rcnn_apples_instance\epoch_12.pth`.

## Entorno conda relacionado

`C:\Users\garci\.conda\envs\openmmlab` — 48.777 archivos, 7.1 GB reales, sin uso desde
enero de 2026. Recreable desde este repo. Pendiente de decidir si se conserva.

---

## Entorno conda `openmmlab` — retirado (2026-08-30)

El entorno `C:\Users\garci\.conda\envs\openmmlab` ocupaba **7.1 GB** y no se usaba
desde el **3 de enero de 2026**. Antes de retirarlo se exportaron dos recetas,
guardadas en esta misma carpeta:

- `openmmlab_environment.yml` — receta completa (conda + pip, con versiones exactas)
- `openmmlab_environment_minimo.yml` — solo los paquetes pedidos explícitamente

Componentes clave que quedaron registrados:

| Paquete | Versión |
|---|---|
| python | 3.8.20 |
| pytorch | 2.1.0 (py3.8_cuda12.1_cudnn8_0) |
| mmcv | 2.1.0 |
| mmengine | 0.10.7 |
| openmim | 0.3.9 |
| pycocotools | 2.0.7 |

### Cómo recrearlo

```bat
conda env create -f openmmlab_environment.yml
conda activate openmmlab
pip install -e .
```

`mmdet` no aparece en la receta porque estaba instalado en modo editable
(`pip install -e .`) desde este mismo repositorio — de ahí el último paso.

### Nota sobre el espacio

Los ~6.5 GB de `C:\Users\garci\.conda\pkgs` son **enlaces duros** a los archivos
del entorno, no una copia aparte (verificado con `fsutil hardlink list`). Por eso
`conda clean --all` no libera nada mientras el entorno exista: hay que eliminar
el entorno primero y limpiar después.
