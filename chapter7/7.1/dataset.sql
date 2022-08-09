create table test01 (
    _id varchar(255) not null primary key,
    Date_Time datetime not null,
    NY_grid double not null,
    NY_EV double not null,
    NY_solar double not null
);
insert into test01 (_id, Date_Time, NY_grid, NY_EV, NY_solar) values ('1', '2019/3/1 0:00', 0.295, 0.003, 0.014);
insert into test01 (_id, Date_Time, NY_grid, NY_EV, NY_solar) values ('2', '2019/3/1 0:15', 0.463, 0.003, 0.014);
insert into test01 (_id, Date_Time, NY_grid, NY_EV, NY_solar) values ('3', '2019/3/1 0:30', 0.366, 0.003, 0.014);
insert into test01 (_id, Date_Time, NY_grid, NY_EV, NY_solar) values ('4', '2019/3/1 0:45', 0.225, 0.003, 0.014);
insert into test01 (_id, Date_Time, NY_grid, NY_EV, NY_solar) values ('5', '2019/3/1 1:00', 0.198, 0.003, 0.014);