import React from 'react';

export interface ConfigEditorProps {
  initialContent: string;
  onChange: (content: string) => void;
  readOnly?: boolean;
}

const ConfigEditor: React.FC<ConfigEditorProps> = ({ initialContent, onChange, readOnly }) => {
  return (
    <textarea
      defaultValue={initialContent}
      readOnly={!!readOnly}
      onChange={e => onChange(e.target.value)}
      rows={12}
      className="input font-mono"
    />
  );
};

export default ConfigEditor;
