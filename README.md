# Budowanie:
cmake -B build
cd build
make

# Uruchamianie wersji na CPU:
./ElectronsAndProtons_CPU
lub
./ElectronsAndProtons_CPU <particle_count>
lub
./ElectronsAndProtons_CPU <particle_count> <width> <height>

# Uruchamianie wersji na GPU:
./ElectronsAndProtons_CUDA
lub
./ElectronsAndProtons_CUDA <particle_count>
lub
./ElectronsAndProtons_CUDA <particle_count> <width> <height>
