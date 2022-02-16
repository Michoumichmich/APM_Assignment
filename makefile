default: all

#export LD_LIBRARY_PATH=$HOME/softs/FreeImage/lib:$LD_LIBRARY_PATH
all:
	nvcc -allow-unsupported-compiler -I${HOME}/softs/FreeImage/include -Iinclude -L${HOME}/softs/FreeImage/lib/ -lfreeimage -std=c++17 -extended-lambda -O3 main.cu src/cuda_filters.cu src/free_image_wrapper.cu -o migdal_main.exe

clean:
	rm -f *.o migdal_main.exe
