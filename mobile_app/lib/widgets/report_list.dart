import 'package:flutter/material.dart';
import 'package:mobile_app/models/report.dart';
import 'package:mobile_app/widgets/report_item.dart';
import 'package:firebase_database/firebase_database.dart';

class ReportList extends StatefulWidget {
  const ReportList({Key? key}) : super(key: key);

  @override
  State<ReportList> createState() => _ReportListState();
}

class _ReportListState extends State<ReportList> {
  final _db = FirebaseDatabase.instance.ref();

  List<Report> _currentReports = [];

  @override
  void initState() {
    super.initState();
    _listeners();
  }

  void _listeners() {
    _db.child('/ClassifiedReports').onValue.listen((event) {
      final values = Map<dynamic, dynamic>.from(
          event.snapshot.value as Map<dynamic, dynamic>);
      List<Report> reportList = [];
      values.entries.forEach((element) {
        reportList.add(
          Report.fromRTDB(Map<dynamic, dynamic>.from(
              element.value as Map<dynamic, dynamic>)),
        );
      });

      reportList.sort(((a, b) {
        var firstDate = a.date;
        var secondDate = b.date;
        return -firstDate.compareTo(secondDate);
      }));

      setState(() {
        _currentReports = reportList;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      child: Expanded(
        child: ListView.builder(
          itemBuilder: (ctx, index) {
            return ReportItem(
              rep: _currentReports[index],
            );
          },
          itemCount: _currentReports.length,
        ),
      ),
    );
  }
}
