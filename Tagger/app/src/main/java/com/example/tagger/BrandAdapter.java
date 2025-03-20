package com.example.tagger;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;

public class BrandAdapter extends BaseAdapter {
    private Context context;
    private String[] brandNames;
    private int[] brandLogos;

    public BrandAdapter(Context context, String[] brandNames, int[] brandLogos) {
        this.context = context;
        this.brandNames = brandNames;
        this.brandLogos = brandLogos;
    }

    @Override
    public int getCount() {
        return brandNames.length;
    }

    @Override
    public Object getItem(int position) {
        return brandNames[position];
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        if (convertView == null) {
            convertView = LayoutInflater.from(context).inflate(R.layout.grid_item, parent, false);
        }

        ImageView brandLogo = convertView.findViewById(R.id.brandLogo);
        brandLogo.setImageResource(brandLogos[position]);

        return convertView;
    }
}
