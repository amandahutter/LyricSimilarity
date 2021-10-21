if  ! command -v wget2 &> /dev/null
then
    echo "wget2 not found on your machine. using wget"
    wget -nc -i ./scripts/urls.txt -P files --progress=bar
    exit
fi

wget2 -nc -i ./scripts/urls.txt -P files --progress=bar