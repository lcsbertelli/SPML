echo "" >fl.log
for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================Round $VARIABLE==========================" 
  python client.py 0 2500 "c1"
  python client.py 2500 5000 "c2"
  python client.py 5000 7500 "c3"
  python client.py 7500 10000 "c4"
  echo "=====================Round $VARIABLE=========================="  >> fl.log
  python server.py >> fl.log 
done
