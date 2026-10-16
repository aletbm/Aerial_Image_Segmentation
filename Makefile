VENV_PY := .venv/bin/python
KERNEL_NAME := final
KERNEL_JSON := $(HOME)/.local/share/jupyter/kernels/$(KERNEL_NAME)/kernel.json
ACTIVATE := .venv/bin/activate
KAGGLE_WIN_PATH := /mnt/c/Users/Alexander/.kaggle/kaggle.json

.PHONY: sync fix-nvrtc ld-path kernel activate-patch fix-kaggle env-file setup train clean-cuda

# 1. Instala/actualiza dependencias del proyecto
sync:
	uv sync

# 2. Crea el symlink que le falta a libnvrtc.so (bug del paquete pip nvidia-cuda-nvrtc-cu11)
fix-nvrtc:
	@$(VENV_PY) -c "\
import glob, os; \
files = glob.glob('.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.*'); \
target = os.path.join(os.path.dirname(files[0]), 'libnvrtc.so') if files else None; \
(os.path.islink(target) or os.path.exists(target)) or os.symlink(os.path.basename(sorted(files)[-1]), target) if files else None; \
print('symlink nvrtc OK' if files else 'ADVERTENCIA: no se encontro libnvrtc.so.*')"

# 3. Calcula el LD_LIBRARY_PATH con todas las libs nvidia del venv (solo imprime, sin side effects)
ld-path:
	@$(VENV_PY) -c "\
import os, sys, glob; \
dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(sys.prefix,'lib','python3.10','site-packages','nvidia','**','*.so*'), recursive=True)}); \
print(':'.join(dirs))"

# 4. Registra el kernel de Jupyter "Python (final)" con LD_LIBRARY_PATH en su env
#    Necesario porque VS Code a veces lanza el kernel via "Jupyter Kernel" (respeta kernel.json)
kernel: fix-nvrtc
	@mkdir -p $(dir $(KERNEL_JSON))
	@$(VENV_PY) -m ipykernel install --user --name=$(KERNEL_NAME) --display-name="Python ($(KERNEL_NAME))"
	@LDPATH=$$($(MAKE) -s ld-path); \
	$(VENV_PY) -c "\
import json; \
path = '$(KERNEL_JSON)'; \
k = json.load(open(path)); \
k['env'] = {'LD_LIBRARY_PATH': '$$LDPATH'}; \
json.dump(k, open(path, 'w'), indent=1)"
	@echo "Kernel Jupyter listo: Python ($(KERNEL_NAME))"

# 5. Parchea .venv/bin/activate para que 'source .venv/bin/activate' deje LD_LIBRARY_PATH listo en bash
activate-patch: fix-nvrtc
	@if ! grep -q "LD_LIBRARY_PATH.*nvidia" $(ACTIVATE) 2>/dev/null; then \
		echo '' >> $(ACTIVATE); \
		echo '# --- fix CUDA libs (agregado por make setup) ---' >> $(ACTIVATE); \
		echo 'export LD_LIBRARY_PATH="$$(python -c \"import os,sys,glob; dirs=sorted({os.path.dirname(p) for p in glob.glob(os.path.join(sys.prefix,\\\"lib\\\",\\\"python3.10\\\",\\\"site-packages\\\",\\\"nvidia\\\",\\\"**\\\",\\\"*.so*\\\"),recursive=True)}); print(\\\":\\\".join(dirs))\"):$$LD_LIBRARY_PATH"' >> $(ACTIVATE); \
		echo "activate parcheado con LD_LIBRARY_PATH"; \
	else \
		echo "activate ya tiene el fix, no se toca"; \
	fi

# 6. Copia kaggle.json desde Windows (WSL) al lado Linux, en ambas rutas posibles
fix-kaggle:
	@if [ -f "$(KAGGLE_WIN_PATH)" ]; then \
		mkdir -p ~/.kaggle ~/.config/kaggle; \
		cp "$(KAGGLE_WIN_PATH)" ~/.kaggle/kaggle.json; \
		cp "$(KAGGLE_WIN_PATH)" ~/.config/kaggle/kaggle.json; \
		chmod 600 ~/.kaggle/kaggle.json ~/.config/kaggle/kaggle.json; \
		echo "kaggle.json copiado desde Windows a ~/.kaggle y ~/.config/kaggle"; \
	else \
		echo "ADVERTENCIA: no se encontro kaggle.json en $(KAGGLE_WIN_PATH)"; \
	fi

# 7. Genera el .env del proyecto (leido por VS Code / Jupyter via "Python Environments")
#    con LD_LIBRARY_PATH. Cubre el caso en que VS Code ignora el kernel.json.
env-file: fix-kaggle
	@LDPATH=$$($(MAKE) -s ld-path); \
	echo "LD_LIBRARY_PATH=$$LDPATH" > .env
	@echo ".env generado con LD_LIBRARY_PATH"

# 8. Todo junto: correr esto despues de clonar el repo o recrear el venv
setup: sync fix-nvrtc kernel activate-patch env-file
	@echo ""
	@echo "Entorno listo."
	@echo "  - En Jupyter/VS Code: selecciona el kernel 'Python (final)'"
	@echo "  - En bash: source .venv/bin/activate"

# 9. Atajo para entrenar desde terminal con las libs CUDA bien seteadas
train: fix-nvrtc
	@LDPATH=$$($(MAKE) -s ld-path); \
	LD_LIBRARY_PATH="$$LDPATH" $(VENV_PY) train_script.py 2>&1 | tee train_log.txt

# 10. Limpieza: borra symlinks/kernels generados, util para probar setup desde cero
clean-cuda:
	rm -f .venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so
	rm -rf $(HOME)/.local/share/jupyter/kernels/$(KERNEL_NAME)
	rm -f .env
	@echo "Limpieza hecha. Corre 'make setup' para reconstruir todo."