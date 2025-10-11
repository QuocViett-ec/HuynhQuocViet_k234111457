import pandas as pd

class MyStatistic:
    def find_orders_within_range(self, df, minValue, maxValue):
        # Tổng giá trị từng đơn hàng
        order_totals = df.groupby('OrderID').apply(
            lambda x: (x['UnitPrice'] * x['Quantity'] * (1 - x['Discount'])).sum()
        )

        # Lọc các đơn hàng có tổng giá trị nằm trong khoảng [minValue, maxValue]
        orders_within_range = order_totals[
            (order_totals >= minValue) & (order_totals <= maxValue)
        ]

        # Danh sách mã đơn hàng duy nhất
        unique_orders = (
            df[df['OrderID'].isin(orders_within_range.index)]['OrderID']
            .drop_duplicates()
            .tolist()
        )

        return unique_orders