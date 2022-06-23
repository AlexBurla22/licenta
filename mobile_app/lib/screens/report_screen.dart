import 'package:flutter/material.dart';
import '../models/report.dart';

class ReportScreen extends StatelessWidget {
  final Report rep;
  const ReportScreen({Key? key, required this.rep}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Raport #${rep.number}'),
      ),
      body: Container(
        color: Theme.of(context).primaryColorLight,
        width: double.infinity,
        child: Column(
          children: [
            Container(
              height: 50,
              child: Text(
                rep.category,
                style: TextStyle(
                  color: Colors.white60,
                  fontSize: 18,
                ),
              ),
              alignment: Alignment.center,
              decoration:
                  BoxDecoration(color: Theme.of(context).primaryColorDark),
            ),
            Container(
              decoration: BoxDecoration(color: Theme.of(context).primaryColor),
              width: double.infinity,
              alignment: Alignment.center,
              padding: EdgeInsets.all(4),
              child: Text(
                rep.link,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.white70,
                ),
              ),
            ),
            Container(
              padding: EdgeInsets.all(5),
              child: Text(rep.date),
            ),
          ],
        ),
      ),
    );
  }
}
