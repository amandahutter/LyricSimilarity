# make sure venv is activated
source bin/activate

# make the build dir if it does not exist
mkdir -p build

wget -nc https://github.com/musixmatch/musixmatch-sdk/raw/master/dist/python-client-generated.zip -P build

unzip ./build/python-client-generated.zip -d ./build -n

cd build/python-client && python3 setup.py install
