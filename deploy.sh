#!/usr/bin/env sh
git stash
git checkout develop
stack build
stack exec site clean
stack exec site build
git fetch --all
git checkout -b master --track origin/master
cp -a _site/ .
git add -A
git commit -m "publish website"
git push origin master
git checkout develop
git branch -D master
git stash pop
