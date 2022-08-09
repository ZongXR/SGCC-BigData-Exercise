select * from test01
into outfile 'test02.csv'
fields terminated by ','
optionally enclosed by '"'
lines terminated by '\n';
