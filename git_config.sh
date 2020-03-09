#recebe args
#$1: endereço para clonar
git init
git clone $1

git config --global user.email "wendersonj@hotmail.com"
git config --global user.name "wendersonj"
git config --global push.default matching
