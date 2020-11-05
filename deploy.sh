#!/usr/bin/env sh
git stash
git checkout develop
git fetch origin master
git checkout -b master --track origin/master
cp -a _site/. .
git add -u .
git commit -m "publish website"
git push origin master
git checkout develop
git branch -D master
git stash pop
