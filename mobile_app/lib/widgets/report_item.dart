import 'package:flutter/material.dart';
import 'package:mobile_app/screens/report_screen.dart';
import '../models/report.dart';

class ReportItem extends StatelessWidget {
  final Report rep;
  const ReportItem({Key? key, required this.rep}) : super(key: key);

  void selectReport(BuildContext ct) {
    Navigator.of(ct).push(
      MaterialPageRoute(
        builder: (_) {
          return ReportScreen(
            rep: rep,
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => selectReport(context),
      child: Card(
        child: Container(
          color: Theme.of(context).primaryColorDark,
          child: Row(
            children: <Widget>[
              Container(
                margin: const EdgeInsets.symmetric(
                  vertical: 10,
                  horizontal: 15,
                ),
                child: Text(
                  'Raport #${rep.number}',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                    color: Colors.white70,
                  ),
                ),
              ),
              Flexible(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      rep.category,
                      softWrap: false,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.bold,
                        color: Theme.of(context).primaryColorLight,
                      ),
                    ),
                    Text(
                      rep.date,
                      style: const TextStyle(color: Colors.white54),
                    ),
                  ],
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}
