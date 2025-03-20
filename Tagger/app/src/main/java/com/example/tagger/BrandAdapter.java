package com.example.tagger;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import androidx.core.content.res.ResourcesCompat;

import java.util.WeakHashMap;

public class BrandAdapter extends BaseAdapter {
    private Context context;
    private String[] brandNames;
    private int[] brandLogos;
    private final WeakHashMap<Integer, Drawable> drawableCache = new WeakHashMap<>();

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
    public boolean hasStableIds() {
        return true;
    }

    static class ViewHolder {
        ImageView brandLogo;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        ViewHolder holder;
        if (convertView == null) {
            convertView = LayoutInflater.from(context).inflate(R.layout.grid_item, parent, false);
            holder = new ViewHolder();
            holder.brandLogo = convertView.findViewById(R.id.brandLogo);
            convertView.setTag(holder);
        } else {
            holder = (ViewHolder) convertView.getTag();
        }

        // Optimizare încărcare imagini
        final int logoResourceId = brandLogos[position];
        
        // Folosim cache pentru a evita încărcarea repetată
        Drawable cachedDrawable = drawableCache.get(position);
        if (cachedDrawable != null) {
            holder.brandLogo.setImageDrawable(cachedDrawable);
        } else {
            try {
                // Înlocuim apelul deprecat getDrawable cu ResourcesCompat
                Drawable drawable = ResourcesCompat.getDrawable(
                    context.getResources(), logoResourceId, context.getTheme());
                
                if (drawable != null) {
                    drawableCache.put(position, drawable);
                    holder.brandLogo.setImageDrawable(drawable);
                } else {
                    // Fallback
                    holder.brandLogo.setImageResource(logoResourceId);
                }
            } catch (Exception e) {
                // Fallback
                holder.brandLogo.setImageResource(logoResourceId);
            }
        }
        
        return convertView;
    }
}
