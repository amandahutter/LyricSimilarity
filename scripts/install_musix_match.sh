# make sure venv is activated
# can't do it here because CI/CD

# make the build dir if it does not exist
mkdir -p ./build

wget -nc https://github.com/musixmatch/musixmatch-sdk/raw/master/dist/python-client-generated.zip -P ./build

unzip -n ./build/python-client-generated.zip -d ./build

cd ./build/python-client && python3 setup.py install
