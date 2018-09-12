# Adesign Makefile

# author: Yuan Chang
# copyright: Copyright (C) 2016-2018
# license: AGPL
# email: pyslvs@gmail.com

all: build

# Into package folder
build: src/*.pyx
ifeq ($(OS),Windows_NT)
	-rename __init__.py .__init__.py
	python setup.py build_ext --inplace
	-rename .__init__.py __init__.py
else
	-mv __init__.py .__init__.py
	python3 setup.py build_ext --inplace
	-mv .__init__.py __init__.py
endif

clean:
ifeq ($(OS),Windows_NT)
	-rename .__init__.py __init__.py
	-del *.pyd /q
	-rd build /s /q
	-del src\*.c /q
	-del src\*.cpp /q
else
	-mv .__init__.py __init__.py
	-rm -f *.so
	-rm -fr build
	-rm -f src/*.c
	-rm -f src/*.cpp
endif
