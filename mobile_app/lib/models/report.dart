class Report {
  final String link;
  final String category;
  final int number;
  final String date;

  Report({
    required this.number,
    required this.link,
    required this.category,
    required this.date,
  });

  factory Report.fromRTDB(Map<dynamic, dynamic> report) {
    return Report(
      number: report['number'] as int,
      link: report['link'] as String,
      category: report['departName'] as String,
      date: report['dateTime'] as String,
    );
  }
}
